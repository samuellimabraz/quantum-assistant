import { NextRequest } from 'next/server';
import { spawn } from 'child_process';
import { writeFile, unlink, mkdir } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';
import { randomUUID } from 'crypto';
import type { TestResult } from '@/types';

export const maxDuration = 60;

interface TestRequestBody {
  userCode: string;
  testCode: string;
  entryPoint: string;
  timeout?: number;
}

/**
 * Build executable code combining solution and test.
 * Matches the logic in sandbox.py for consistent behavior.
 */
function buildExecutableCode(
  userCode: string,
  testCode: string,
  entryPoint: string
): string {
  // Deduplicate imports between code and test
  const codeImports = new Set(
    userCode.match(/^(?:from|import)\s+.+$/gm) || []
  );

  const testLines: string[] = [];
  for (const line of testCode.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.startsWith('from ') || trimmed.startsWith('import ')) {
      if (!codeImports.has(trimmed)) {
        testLines.push(line);
      }
    } else {
      testLines.push(line);
    }
  }

  const cleanedTest = testLines.join('\n');

  // Determine test execution trigger (matching sandbox.py logic)
  const executionTrigger = getTestExecutionTrigger(testCode, entryPoint);

  return `${userCode}

${cleanedTest}${executionTrigger}

print("TEST_PASSED")
`;
}

/**
 * Determine the test execution trigger.
 * Matches the generation phase logic for invoking check() or test_* functions.
 */
function getTestExecutionTrigger(testCode: string, entryPoint: string): string {
  const hasCheck = /def\s+check\s*\(/.test(testCode);
  const testFuncMatch = testCode.match(/def\s+(test_\w+)\s*\(/);

  if (hasCheck && entryPoint) {
    // Check if check() is already called with entry_point
    const checkCallPattern = new RegExp(
      `check\\s*\\(\\s*${entryPoint.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*\\)`
    );
    if (checkCallPattern.test(testCode)) {
      return '';
    }
    return `\ncheck(${entryPoint})`;
  } else if (testFuncMatch) {
    const testName = testFuncMatch[1];
    return `\n${testName}()`;
  }

  return '';
}

/**
 * Extract meaningful error message from stderr.
 * Matches the logic in sandbox.py for better error reporting.
 */
function extractErrorMessage(stderr: string): string {
  if (!stderr) return 'Unknown error';

  const lines = stderr.split('\n');

  // Error types to look for
  const errorTypes = [
    'AssertionError',
    'TypeError',
    'ValueError',
    'AttributeError',
    'ImportError',
    'ModuleNotFoundError',
    'NameError',
    'KeyError',
    'IndexError',
    'RuntimeError',
    'SyntaxError',
    'IndentationError',
  ];

  // Find the error line (last line starting with a known error type)
  let errorLineIdx = -1;
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    if (
      line &&
      (errorTypes.some((et) => line.startsWith(et)) || line.includes('Error:'))
    ) {
      errorLineIdx = i;
      break;
    }
  }

  if (errorLineIdx === -1) {
    return stderr.slice(-500).trim();
  }

  const errorLine = lines[errorLineIdx].trim();

  // For AssertionError, find the assertion line that failed
  if (errorLine.startsWith('AssertionError')) {
    // Look backwards for the assertion statement
    for (let i = errorLineIdx - 1; i >= Math.max(0, errorLineIdx - 10); i--) {
      const line = lines[i].trim();
      if (line.startsWith('assert ')) {
        if (errorLine === 'AssertionError') {
          return `AssertionError at: ${line}`;
        }
        return `${errorLine} at: ${line}`;
      }
    }

    // If no assert found, look for the "File" line with context
    for (let i = errorLineIdx - 1; i >= Math.max(0, errorLineIdx - 5); i--) {
      if (lines[i].includes('File ') && lines[i].includes(', line ')) {
        if (i + 1 < errorLineIdx) {
          const codeLine = lines[i + 1].trim();
          if (errorLine === 'AssertionError') {
            return `AssertionError at: ${codeLine}`;
          }
          return `${errorLine} at: ${codeLine}`;
        }
        break;
      }
    }
  }

  // For AttributeError/ImportError, include full message
  if (
    errorLine.startsWith('AttributeError') ||
    errorLine.startsWith('ImportError') ||
    errorLine.startsWith('ModuleNotFoundError')
  ) {
    return errorLine;
  }

  return errorLine || stderr.slice(-500).trim();
}

/**
 * Run tests using subprocess execution matching the evaluate module behavior.
 * Uses the same execution model as sandbox.py for consistency.
 */
async function runTests(
  userCode: string,
  testCode: string,
  entryPoint: string,
  timeout: number
): Promise<TestResult> {
  const startTime = Date.now();
  const tempDir = join(tmpdir(), 'quantum-sandbox');
  const tempFile = join(tempDir, `test_${randomUUID()}.py`);

  try {
    await mkdir(tempDir, { recursive: true });

    // Build executable code matching sandbox.py logic
    const fullCode = buildExecutableCode(userCode, testCode, entryPoint);
    await writeFile(tempFile, fullCode, 'utf-8');

    return await new Promise<TestResult>((resolve) => {
      let stdout = '';
      let stderr = '';
      let killed = false;

      // Use the PYTHON_PATH environment variable if set, otherwise default to python3
      // This allows configuring the Python environment with quantum dependencies
      const pythonPath = process.env.PYTHON_PATH || 'python3';

      const pythonProcess = spawn(pythonPath, [tempFile], {
        timeout: timeout * 1000,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1',
          MPLBACKEND: 'Agg',
          MallocStackLogging: '0',
          MallocNanoZone: '0',
        },
      });

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      const timeoutId = setTimeout(() => {
        killed = true;
        pythonProcess.kill('SIGKILL');
      }, timeout * 1000);

      pythonProcess.on('close', (code) => {
        clearTimeout(timeoutId);
        const executionTime = Date.now() - startTime;

        if (killed) {
          resolve({
            passed: false,
            total: 1,
            failed: 1,
            details: [
              {
                name: 'Execution',
                passed: false,
                error: `Execution timeout (>${timeout}s). Your code took too long to execute.`,
              },
            ],
            executionTime,
            error: `Execution timeout (>${timeout}s)`,
          });
          return;
        }

        // Clean up stdout and check for TEST_PASSED marker
        const stdoutClean = stdout.trim();
        const testPassed = code === 0 && stdoutClean.includes('TEST_PASSED');

        if (testPassed) {
          // Success - all tests passed
          // Include any output before TEST_PASSED (useful debugging info)
          const outputBeforePass = stdoutClean
            .replace('TEST_PASSED', '')
            .trim();

          resolve({
            passed: true,
            total: 1,
            failed: 0,
            details: [
              {
                name: 'All tests',
                passed: true,
              },
            ],
            executionTime,
            output: outputBeforePass || undefined,
          });
        } else {
          // Failure - extract meaningful error message
          const cleanStderr = stderr
            .split('\n')
            .filter(
              (line) =>
                !line.includes('UserWarning') &&
                !line.includes('DeprecationWarning') &&
                !line.includes('FutureWarning') &&
                !line.includes('from cryptography')
            )
            .join('\n')
            .trim();

          const errorMessage = extractErrorMessage(cleanStderr);

          // Include full traceback for debugging
          const fullTraceback = cleanStderr || stderr.trim();

          resolve({
            passed: false,
            total: 1,
            failed: 1,
            details: [
              {
                name: 'Test execution',
                passed: false,
                error: errorMessage,
              },
            ],
            executionTime,
            error: errorMessage,
            traceback: fullTraceback !== errorMessage ? fullTraceback : undefined,
            output: stdoutClean || undefined,
          });
        }
      });

      pythonProcess.on('error', (err) => {
        clearTimeout(timeoutId);
        resolve({
          passed: false,
          total: 0,
          failed: 0,
          details: [],
          executionTime: Date.now() - startTime,
          error: `Failed to start Python: ${err.message}`,
        });
      });
    });
  } finally {
    try {
      await unlink(tempFile);
    } catch {
      // Ignore cleanup errors
    }
  }
}

export async function POST(request: NextRequest) {
  try {
    const body: TestRequestBody = await request.json();
    const { userCode, testCode, entryPoint, timeout = 30 } = body;

    if (!userCode || typeof userCode !== 'string') {
      return new Response(
        JSON.stringify({ error: 'Invalid request: userCode string required' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    if (!testCode || typeof testCode !== 'string') {
      return new Response(
        JSON.stringify({ error: 'Invalid request: testCode string required' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Limit code length
    if (userCode.length > 50000 || testCode.length > 50000) {
      return new Response(
        JSON.stringify({ error: 'Code too long (max 50KB each)' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    const safeTimeout = Math.min(Math.max(timeout, 5), 60);
    const result = await runTests(userCode, testCode, entryPoint, safeTimeout);

    return new Response(JSON.stringify(result), {
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Test API error:', error);

    return new Response(
      JSON.stringify({
        passed: false,
        total: 0,
        failed: 0,
        details: [],
        executionTime: 0,
        error: error instanceof Error ? error.message : 'Test execution failed',
      } as TestResult),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}

