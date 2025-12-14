import { NextRequest } from 'next/server';
import { spawn } from 'child_process';
import { writeFile, unlink, mkdir } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';
import { randomUUID } from 'crypto';

export const maxDuration = 60;

interface ExecuteRequestBody {
    code: string;
    timeout?: number;
}

interface ExecutionResult {
    success: boolean;
    output: string;
    error: string;
    executionTime: number;
    hasCircuitOutput: boolean;
}

// Safety wrapper that captures output and prevents dangerous operations
// More permissive than before - allows Qiskit cache writes but blocks user directory writes
const SAFETY_WRAPPER = `
import sys
import io
import os
import warnings
from contextlib import redirect_stdout, redirect_stderr

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Capture all output
_stdout_capture = io.StringIO()
_stderr_capture = io.StringIO()

# Block dangerous system operations but allow file I/O for Qiskit
os.system = lambda *args, **kwargs: None
os.popen = lambda *args, **kwargs: None

try:
    import subprocess
    subprocess.run = lambda *args, **kwargs: None
    subprocess.call = lambda *args, **kwargs: None
    subprocess.Popen = lambda *args, **kwargs: None
except ImportError:
    pass

# Now execute the user code with output capture
with redirect_stdout(_stdout_capture), redirect_stderr(_stderr_capture):
    try:
        exec(compile('''
__USER_CODE__
''', '<user_code>', 'exec'), globals())
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)

# Print captured output
_stdout_result = _stdout_capture.getvalue()
_stderr_result = _stderr_capture.getvalue()

if _stdout_result:
    print(_stdout_result, end='')
if _stderr_result:
    print(_stderr_result, end='', file=sys.stderr)
`;

function createSafeCode(userCode: string): string {
    // Escape the user code for embedding
    const escapedCode = userCode
        .replace(/\\/g, '\\\\')
        .replace(/'''/g, "\\'\\'\\'");

    return SAFETY_WRAPPER.replace('__USER_CODE__', escapedCode);
}

function detectCircuitOutput(code: string, output: string): boolean {
    // Check if the code likely produces circuit visualization
    const circuitPatterns = [
        /\.draw\(/,
        /circuit_drawer/,
        /plot_histogram/,
        /plot_bloch/,
        /\.decompose\(\)/,
        /print.*circuit/i,
    ];

    const outputPatterns = [
        /[┌─┬┐│├┼┤└┴┘═║╔╗╚╝]/,  // ASCII circuit characters
        /q\d*.*[─┤├]/,           // Qubit lines
        /[HXYZTSRx].*├/,         // Gate symbols
    ];

    const hasCircuitCode = circuitPatterns.some(p => p.test(code));
    const hasCircuitOutput = outputPatterns.some(p => p.test(output));

    return hasCircuitCode || hasCircuitOutput;
}

async function executeCode(code: string, timeout: number): Promise<ExecutionResult> {
    const startTime = Date.now();
    const tempDir = join(tmpdir(), 'quantum-sandbox');
    const tempFile = join(tempDir, `exec_${randomUUID()}.py`);

    try {
        // Ensure temp directory exists
        await mkdir(tempDir, { recursive: true });

        // Create safe wrapped code
        const safeCode = createSafeCode(code);
        await writeFile(tempFile, safeCode, 'utf-8');

        return await new Promise<ExecutionResult>((resolve) => {
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
                    MPLBACKEND: 'Agg',  // Non-interactive matplotlib
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

            pythonProcess.on('close', (exitCode) => {
                clearTimeout(timeoutId);
                const executionTime = Date.now() - startTime;

                if (killed) {
                    resolve({
                        success: false,
                        output: stdout,
                        error: `Execution timeout (>${timeout}s). The code took too long to execute.`,
                        executionTime,
                        hasCircuitOutput: false,
                    });
                    return;
                }

                // Clean up stderr from common warnings
                const cleanStderr = stderr
                    .split('\n')
                    .filter(line => !line.includes('UserWarning') &&
                        !line.includes('DeprecationWarning') &&
                        !line.includes('FutureWarning'))
                    .join('\n')
                    .trim();

                const success = exitCode === 0 && !cleanStderr;

                resolve({
                    success,
                    output: stdout.trim(),
                    error: cleanStderr,
                    executionTime,
                    hasCircuitOutput: detectCircuitOutput(code, stdout),
                });
            });

            pythonProcess.on('error', (err) => {
                clearTimeout(timeoutId);
                resolve({
                    success: false,
                    output: '',
                    error: `Failed to start Python: ${err.message}`,
                    executionTime: Date.now() - startTime,
                    hasCircuitOutput: false,
                });
            });
        });
    } finally {
        // Clean up temp file
        try {
            await unlink(tempFile);
        } catch {
            // Ignore cleanup errors
        }
    }
}

export async function POST(request: NextRequest) {
    try {
        const body: ExecuteRequestBody = await request.json();
        const { code, timeout = 30 } = body;

        if (!code || typeof code !== 'string') {
            return new Response(
                JSON.stringify({ error: 'Invalid request: code string required' }),
                { status: 400, headers: { 'Content-Type': 'application/json' } }
            );
        }

        // Limit code length
        if (code.length > 50000) {
            return new Response(
                JSON.stringify({ error: 'Code too long (max 50KB)' }),
                { status: 400, headers: { 'Content-Type': 'application/json' } }
            );
        }

        // Limit timeout
        const safeTimeout = Math.min(Math.max(timeout, 5), 60);

        const result = await executeCode(code, safeTimeout);

        return new Response(JSON.stringify(result), {
            headers: { 'Content-Type': 'application/json' },
        });
    } catch (error) {
        console.error('Execute API error:', error);

        return new Response(
            JSON.stringify({
                success: false,
                output: '',
                error: error instanceof Error ? error.message : 'Execution failed',
                executionTime: 0,
                hasCircuitOutput: false,
            }),
            { status: 500, headers: { 'Content-Type': 'application/json' } }
        );
    }
}

