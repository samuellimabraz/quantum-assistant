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

// List of dangerous patterns that should be blocked (same as execute route)
const DANGEROUS_PATTERNS = [
  // Environment variable access
  /os\.environ/,
  /environ\[/,
  /getenv\s*\(/,
  // Dangerous modules
  /\bctypes\b/,
  /\bpickle\b/,
  /\bmarshal\b/,
  /\bshelve\b/,
  /\bcommands\b/,
  /\bpty\b/,
  /\bpexpect\b/,
  // System/shell access
  /\bos\.system\b/,
  /\bos\.popen\b/,
  /\bos\.spawn/,
  /\bos\.exec/,
  /\bos\.fork\b/,
  /\bsubprocess\b/,
  /\bcommands\b/,
  // File system attacks outside sandbox
  /open\s*\(\s*['"]\s*\/etc/,
  /open\s*\(\s*['"]\s*\/proc/,
  /open\s*\(\s*['"]\s*\/sys/,
  /open\s*\(\s*['"]\s*\/dev/,
  /open\s*\(\s*['"]\s*\/var/,
  /open\s*\(\s*['"]\s*\/root/,
  /open\s*\(\s*['"]\s*\/home/,
  /open\s*\(\s*['"]\s*\/tmp/,
  /open\s*\(\s*['"]\s*\.env/,
  /open\s*\(\s*['"]\s*\.\.\//, // Path traversal
  /open\s*\(\s*f?['"]\s*\{/,  // f-string with path
  // Network access
  /\bsocket\b/,
  /\burllib\b/,
  /\brequests\b/,
  /\bhttpx\b/,
  /\baiohttp\b/,
  /\bhttp\.client\b/,
  /\bftplib\b/,
  /\bsmtplib\b/,
  /\btelnetlib\b/,
  /\bparamiko\b/,
  // Code execution
  /\beval\s*\(/,
  /\bexec\s*\(/,
  /\bcompile\s*\(/,
  /\b__import__\b/,
  /\bimportlib\b/,
  /\bbuiltins\b/,
  /\bglobals\s*\(\s*\)/,
  /\blocals\s*\(\s*\)/,
  /\bgetattr\s*\([^,]+,\s*['"]/,  // getattr with string
  /\bsetattr\s*\(/,
  /\bdelattr\s*\(/,
  // Class/object manipulation for sandbox escape
  /\b__class__\b/,
  /\b__bases__\b/,
  /\b__subclasses__\b/,
  /\b__mro__\b/,
  /\b__globals__\b/,
  /\b__code__\b/,
  /\b__reduce__\b/,
  /\b__getstate__\b/,
  /\b__setstate__\b/,
  // Multiprocessing (can be used to bypass restrictions)
  /\bmultiprocessing\b/,
  /\bthreading\b/,
  /\bconcurrent\b/,
  /\basyncio\.subprocess/,
];

function validateCode(code: string): { valid: boolean; error?: string } {
  const codeWithoutComments = code
    .replace(/#.*$/gm, '')  // Remove single-line comments
    .replace(/'''[\s\S]*?'''/g, '')  // Remove triple-single-quote strings
    .replace(/"""[\s\S]*?"""/g, ''); // Remove triple-double-quote strings

  for (const pattern of DANGEROUS_PATTERNS) {
    if (pattern.test(codeWithoutComments)) {
      return {
        valid: false,
        error: `Security error: Potentially dangerous code pattern detected. For security reasons, certain operations are not allowed in the sandbox.`
      };
    }
  }

  return { valid: true };
}

// Minimal safe environment variables for Python execution
function getSafeEnv(): Record<string, string> {
  const env: Record<string, string> = {
    PATH: '/usr/bin:/bin:/usr/local/bin',
    HOME: '/tmp',
    PYTHONUNBUFFERED: '1',
    MPLBACKEND: 'Agg',
    MallocStackLogging: '0',
    MallocNanoZone: '0',
    LANG: 'en_US.UTF-8',
    LC_ALL: 'en_US.UTF-8',
  };
  // Only pass PYTHON_PATH if needed, but not other secrets
  if (process.env.PYTHON_PATH) {
    env.PYTHON_PATH = process.env.PYTHON_PATH;
  }
  return env;
}

// Security wrapper for test execution
// Primary security is pattern detection + clean environment
const SECURITY_WRAPPER = `
import sys
import io
import os
import builtins
import warnings
from contextlib import redirect_stdout, redirect_stderr

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================
# SECURITY SANDBOX SETUP (Second Line of Defense)
# Primary security is pattern detection + clean environment
# ============================================

# Block dangerous system operations
os.system = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("os.system not allowed in sandbox"))
os.popen = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("os.popen not allowed in sandbox"))
if hasattr(os, 'spawn'):
    os.spawn = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("os.spawn not allowed"))
if hasattr(os, 'spawnl'):
    os.spawnl = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("os.spawnl not allowed"))
if hasattr(os, 'spawnle'):
    os.spawnle = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("os.spawnle not allowed"))
if hasattr(os, 'spawnlp'):
    os.spawnlp = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("os.spawnlp not allowed"))
if hasattr(os, 'spawnlpe'):
    os.spawnlpe = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("os.spawnlpe not allowed"))
if hasattr(os, 'spawnv'):
    os.spawnv = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("os.spawnv not allowed"))
if hasattr(os, 'spawnve'):
    os.spawnve = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("os.spawnve not allowed"))
if hasattr(os, 'spawnvp'):
    os.spawnvp = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("os.spawnvp not allowed"))
if hasattr(os, 'spawnvpe'):
    os.spawnvpe = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("os.spawnvpe not allowed"))
if hasattr(os, 'execl'):
    os.execl = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("os.execl not allowed"))
if hasattr(os, 'execle'):
    os.execle = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("os.execle not allowed"))
if hasattr(os, 'execlp'):
    os.execlp = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("os.execlp not allowed"))
if hasattr(os, 'execlpe'):
    os.execlpe = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("os.execlpe not allowed"))
if hasattr(os, 'execv'):
    os.execv = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("os.execv not allowed"))
if hasattr(os, 'execve'):
    os.execve = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("os.execve not allowed"))
if hasattr(os, 'execvp'):
    os.execvp = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("os.execvp not allowed"))
if hasattr(os, 'execvpe'):
    os.execvpe = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("os.execvpe not allowed"))
if hasattr(os, 'fork'):
    os.fork = lambda: (_ for _ in ()).throw(PermissionError("os.fork not allowed"))
if hasattr(os, 'forkpty'):
    os.forkpty = lambda: (_ for _ in ()).throw(PermissionError("os.forkpty not allowed"))
if hasattr(os, 'killpg'):
    os.killpg = lambda *args: (_ for _ in ()).throw(PermissionError("os.killpg not allowed"))
if hasattr(os, 'kill'):
    os.kill = lambda *args: (_ for _ in ()).throw(PermissionError("os.kill not allowed"))

# Block subprocess module
try:
    import subprocess
    subprocess.run = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("subprocess not allowed"))
    subprocess.call = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("subprocess not allowed"))
    subprocess.check_call = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("subprocess not allowed"))
    subprocess.check_output = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("subprocess not allowed"))
    subprocess.Popen = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("subprocess not allowed"))
    subprocess.getoutput = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("subprocess not allowed"))
    subprocess.getstatusoutput = lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("subprocess not allowed"))
except ImportError:
    pass

# Create restricted open function to block access to sensitive files
_original_open = builtins.open
_ALLOWED_PATHS = ['/tmp/quantum-sandbox']

def _restricted_open(file, mode='r', *args, **kwargs):
    """Restricted open that blocks access to sensitive files"""
    if isinstance(file, (str, bytes)):
        file_str = file if isinstance(file, str) else file.decode()
        if file_str.startswith('/'):
            file_str_lower = file_str.lower()
            
            # Block reading system sensitive paths
            blocked_prefixes = ['/etc/passwd', '/etc/shadow', '/proc/self', '/proc/1']
            for prefix in blocked_prefixes:
                if file_str_lower.startswith(prefix):
                    raise PermissionError(f"Access to {prefix} is not allowed in sandbox")
            
            # Block reading obvious secrets
            blocked_patterns = ['.env.local', '.env.', 'secrets', 'credentials', 'private_key']
            for pattern in blocked_patterns:
                if pattern in file_str_lower:
                    raise PermissionError(f"Access to files matching '{pattern}' is not allowed in sandbox")
    
    return _original_open(file, mode, *args, **kwargs)

builtins.open = _restricted_open

# ============================================
# END SECURITY SANDBOX SETUP
# ============================================

# Setup matplotlib non-interactive backend
try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

# Now execute the user code
`;

/**
 * Build executable code combining solution and test with security wrapper.
 */
function buildSecureExecutableCode(
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
  const executionTrigger = getTestExecutionTrigger(testCode, entryPoint);

  // Escape user code for embedding
  const escapedUserCode = userCode
    .replace(/\\/g, '\\\\')
    .replace(/'''/g, "\\'\\'\\'");
  
  const escapedTestCode = cleanedTest
    .replace(/\\/g, '\\\\')
    .replace(/'''/g, "\\'\\'\\'");

  return `${SECURITY_WRAPPER}
try:
    exec(compile('''
${escapedUserCode}

${escapedTestCode}${executionTrigger}

print("TEST_PASSED")
''', '<user_code>', 'exec'), {'__builtins__': builtins, '__name__': '__main__'})
except Exception as e:
    import traceback
    traceback.print_exc()
`;
}

/**
 * Determine the test execution trigger.
 */
function getTestExecutionTrigger(testCode: string, entryPoint: string): string {
  const hasCheck = /def\s+check\s*\(/.test(testCode);
  const testFuncMatch = testCode.match(/def\s+(test_\w+)\s*\(/);

  if (hasCheck && entryPoint) {
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
 */
function extractErrorMessage(stderr: string): string {
  if (!stderr) return 'Unknown error';

  const lines = stderr.split('\n');

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
    'PermissionError',
  ];

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

  if (errorLine.startsWith('AssertionError')) {
    for (let i = errorLineIdx - 1; i >= Math.max(0, errorLineIdx - 10); i--) {
      const line = lines[i].trim();
      if (line.startsWith('assert ')) {
        if (errorLine === 'AssertionError') {
          return `AssertionError at: ${line}`;
        }
        return `${errorLine} at: ${line}`;
      }
    }

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

  if (
    errorLine.startsWith('AttributeError') ||
    errorLine.startsWith('ImportError') ||
    errorLine.startsWith('ModuleNotFoundError') ||
    errorLine.startsWith('PermissionError')
  ) {
    return errorLine;
  }

  return errorLine || stderr.slice(-500).trim();
}

/**
 * Run tests with security wrapper.
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

    // Build secure executable code with security wrapper
    const fullCode = buildSecureExecutableCode(userCode, testCode, entryPoint);
    await writeFile(tempFile, fullCode, 'utf-8');

    return await new Promise<TestResult>((resolve) => {
      let stdout = '';
      let stderr = '';
      let killed = false;

      const pythonPath = process.env.PYTHON_PATH || 'python3';

      const pythonProcess = spawn(pythonPath, [tempFile], {
        timeout: timeout * 1000,
        env: getSafeEnv() as NodeJS.ProcessEnv,  // Use minimal safe environment
        cwd: tempDir,  // Run in isolated temp directory
      });

      pythonProcess.stdout.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data: Buffer) => {
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

        const stdoutClean = stdout.trim();
        const testPassed = code === 0 && stdoutClean.includes('TEST_PASSED');

        if (testPassed) {
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

    // Validate user code for dangerous patterns (first line of defense)
    const userValidation = validateCode(userCode);
    if (!userValidation.valid) {
      return new Response(
        JSON.stringify({
          passed: false,
          total: 0,
          failed: 0,
          details: [],
          executionTime: 0,
          error: userValidation.error,
        } as TestResult),
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
