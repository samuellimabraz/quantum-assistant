import { NextRequest } from 'next/server';
import { spawn } from 'child_process';
import { writeFile, unlink, mkdir, readFile, readdir } from 'fs/promises';
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
    images?: string[]; // Base64 encoded images
}

const DANGEROUS_PATTERNS = [
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

const ALLOWED_PATTERNS = [
    /from qiskit/,
    /import qiskit/,
    /from numpy/,
    /import numpy/,
    /from scipy/,
    /import scipy/,
    /from matplotlib/,
    /import matplotlib/,
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

const createSafetyWrapper = (figureDir: string) => `
import sys
import io
import os
import warnings
import builtins
from contextlib import redirect_stdout, redirect_stderr

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup figure capture directory
_FIGURE_DIR = "${figureDir.replace(/\\/g, '\\\\')}"
_figure_counter = [0]

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
_ALLOWED_PATHS = [_FIGURE_DIR, '/tmp/quantum-sandbox']

def _restricted_open(file, mode='r', *args, **kwargs):
    """Restricted open that blocks access to sensitive files"""
    if isinstance(file, (str, bytes)):
        file_str = file if isinstance(file, str) else file.decode()
        # Don't block relative paths that are needed for library operation
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

# Capture all output
_stdout_capture = io.StringIO()
_stderr_capture = io.StringIO()

# Setup matplotlib figure capture
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    _original_show = plt.show
    _original_savefig = plt.savefig
    
    def _capture_show(*args, **kwargs):
        """Capture plt.show() calls and save figures"""
        figs = [plt.figure(i) for i in plt.get_fignums()]
        for fig in figs:
            _figure_counter[0] += 1
            filepath = os.path.join(_FIGURE_DIR, f"figure_{_figure_counter[0]}.png")
            fig.savefig(filepath, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='#18181b', edgecolor='none', transparent=False)
        plt.close('all')
    
    def _capture_savefig(fname, *args, **kwargs):
        """Capture savefig calls"""
        _figure_counter[0] += 1
        filepath = os.path.join(_FIGURE_DIR, f"figure_{_figure_counter[0]}.png")
        kwargs_copy = dict(kwargs)
        kwargs_copy['format'] = 'png'
        kwargs_copy['dpi'] = kwargs_copy.get('dpi', 150)
        kwargs_copy['bbox_inches'] = kwargs_copy.get('bbox_inches', 'tight')
        kwargs_copy['facecolor'] = kwargs_copy.get('facecolor', '#18181b')
        _original_savefig(filepath, **kwargs_copy)
    
    plt.show = _capture_show
    plt.savefig = _capture_savefig
    
    # Also capture Qiskit circuit.draw() with mpl output
    try:
        from qiskit import QuantumCircuit
        _original_draw = QuantumCircuit.draw
        
        def _capture_draw(self, output=None, **kwargs):
            result = _original_draw(self, output=output, **kwargs)
            if output == 'mpl' and result is not None:
                _figure_counter[0] += 1
                filepath = os.path.join(_FIGURE_DIR, f"figure_{_figure_counter[0]}.png")
                result.savefig(filepath, format='png', dpi=150, bbox_inches='tight',
                              facecolor='#18181b', edgecolor='none')
                plt.close(result)
            return result
        
        QuantumCircuit.draw = _capture_draw
    except ImportError:
        pass
        
except ImportError:
    pass

# Now execute the user code with output capture
with redirect_stdout(_stdout_capture), redirect_stderr(_stderr_capture):
    try:
        exec(compile('''
__USER_CODE__
''', '<user_code>', 'exec'), {'__builtins__': builtins, '__name__': '__main__'})
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)

# Final figure capture - save any remaining open figures
try:
    import matplotlib.pyplot as plt
    figs = [plt.figure(i) for i in plt.get_fignums()]
    for fig in figs:
        _figure_counter[0] += 1
        filepath = os.path.join(_FIGURE_DIR, f"figure_{_figure_counter[0]}.png")
        fig.savefig(filepath, format='png', dpi=150, bbox_inches='tight',
                   facecolor='#18181b', edgecolor='none', transparent=False)
    plt.close('all')
except:
    pass

# Print captured output
_stdout_result = _stdout_capture.getvalue()
_stderr_result = _stderr_capture.getvalue()

if _stdout_result:
    print(_stdout_result, end='')
if _stderr_result:
    print(_stderr_result, end='', file=sys.stderr)
`;

function createSafeCode(userCode: string, figureDir: string): string {
    const escapedCode = userCode
        .replace(/\\/g, '\\\\')
        .replace(/'''/g, "\\'\\'\\'");

    return createSafetyWrapper(figureDir).replace('__USER_CODE__', escapedCode);
}

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
    if (process.env.PYTHON_PATH) {
        env.PYTHON_PATH = process.env.PYTHON_PATH;
    }
    return env;
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

async function collectFigures(figureDir: string): Promise<string[]> {
    const images: string[] = [];
    try {
        const files = await readdir(figureDir);
        const pngFiles = files.filter(f => f.endsWith('.png')).sort();

        for (const file of pngFiles) {
            const filepath = join(figureDir, file);
            const data = await readFile(filepath);
            images.push(data.toString('base64'));
            // Clean up the file
            await unlink(filepath).catch(() => { });
        }
    } catch {
        // Directory might not exist or be empty
    }
    return images;
}

async function executeCode(code: string, timeout: number): Promise<ExecutionResult> {
    const startTime = Date.now();
    const execId = randomUUID();
    const tempDir = join(tmpdir(), 'quantum-sandbox');
    const figureDir = join(tempDir, `figures_${execId}`);
    const tempFile = join(tempDir, `exec_${execId}.py`);

    try {
        // Ensure temp directories exist
        await mkdir(tempDir, { recursive: true });
        await mkdir(figureDir, { recursive: true });

        // Create safe wrapped code
        const safeCode = createSafeCode(code, figureDir);
        await writeFile(tempFile, safeCode, 'utf-8');

        return await new Promise<ExecutionResult>(async (resolve) => {
            let stdout = '';
            let stderr = '';
            let killed = false;

            // Use the PYTHON_PATH environment variable if set, otherwise default to python3
            const pythonPath = process.env.PYTHON_PATH || 'python3';

            const pythonProcess = spawn(pythonPath, [tempFile], {
                timeout: timeout * 1000,
                env: getSafeEnv() as NodeJS.ProcessEnv,  // Use minimal safe environment, no secrets
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

            pythonProcess.on('close', async (exitCode: number | null) => {
                clearTimeout(timeoutId);
                const executionTime = Date.now() - startTime;

                // Collect any generated figures
                const images = await collectFigures(figureDir);

                if (killed) {
                    resolve({
                        success: false,
                        output: stdout,
                        error: `Execution timeout (>${timeout}s). The code took too long to execute.`,
                        executionTime,
                        hasCircuitOutput: false,
                        images,
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
                    images,
                });
            });

            pythonProcess.on('error', (err: Error) => {
                clearTimeout(timeoutId);
                resolve({
                    success: false,
                    output: '',
                    error: `Failed to start Python: ${err.message}`,
                    executionTime: Date.now() - startTime,
                    hasCircuitOutput: false,
                    images: [],
                });
            });
        });
    } finally {
        // Clean up temp file and figure directory
        try {
            await unlink(tempFile);
        } catch {
            // Ignore cleanup errors
        }
        try {
            // Clean up figure directory
            const files = await readdir(figureDir).catch(() => []);
            for (const file of files) {
                await unlink(join(figureDir, file)).catch(() => { });
            }
            await unlink(figureDir).catch(() => { });
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

        // Validate code for dangerous patterns (first line of defense)
        const validation = validateCode(code);
        if (!validation.valid) {
            return new Response(
                JSON.stringify({
                    success: false,
                    output: '',
                    error: validation.error,
                    executionTime: 0,
                    hasCircuitOutput: false,
                    images: [],
                }),
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
                images: [],
            }),
            { status: 500, headers: { 'Content-Type': 'application/json' } }
        );
    }
}
