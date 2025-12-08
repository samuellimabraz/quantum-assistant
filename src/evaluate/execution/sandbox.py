"""Safe code execution sandbox for evaluating generated code.

Uses subprocess execution to match the generation phase behavior,
ensuring imports and module resolution work identically.
"""

import asyncio
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _get_subprocess_env() -> dict[str, str]:
    """Get environment for subprocess execution.

    Copies current environment and disables macOS malloc logging
    to avoid noise in test output.
    """
    env = os.environ.copy()
    env["MallocStackLogging"] = "0"
    env["MallocNanoZone"] = "0"
    return env


def _load_dotenv() -> None:
    """Load .env file for environment variables like QISKIT_IBM_TOKEN."""
    try:
        from dotenv import load_dotenv

        current = Path.cwd()
        for parent in [current] + list(current.parents):
            env_file = parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                break
    except ImportError:
        pass


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    output: str = ""
    error: str = ""
    timeout: bool = False
    return_value: Any = None


class CodeExecutor:
    """Execute code safely using subprocess.

    Uses the same execution model as the generation phase (asyncio subprocess)
    to ensure consistent import resolution and module loading.
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize code executor.

        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout
        _load_dotenv()

    def execute(
        self,
        code: str,
        test_code: str | None = None,
        entry_point: str | None = None,
    ) -> ExecutionResult:
        """
        Execute code with optional test.

        Args:
            code: The code to execute
            test_code: Optional test code to run
            entry_point: Optional entry point function name

        Returns:
            ExecutionResult with execution status and output
        """
        return asyncio.run(self.execute_async(code, test_code, entry_point))

    async def execute_async(
        self,
        code: str,
        test_code: str | None = None,
        entry_point: str | None = None,
    ) -> ExecutionResult:
        """
        Execute code with optional test asynchronously.

        Uses subprocess execution matching the generation phase approach.
        """
        combined_code = self._build_executable_code(code, test_code, entry_point)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(combined_code)
            temp_path = f.name

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_get_subprocess_env(),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ExecutionResult(
                    success=False,
                    error=f"Execution timeout (>{self.timeout}s)",
                    timeout=True,
                )

            stdout_text = stdout.decode("utf-8").strip()
            stderr_text = stderr.decode("utf-8").strip()

            if process.returncode == 0 and "TEST_PASSED" in stdout_text:
                return ExecutionResult(success=True, output=stdout_text)

            error_message = self._extract_error_message(stderr_text)
            return ExecutionResult(
                success=False,
                output=stdout_text,
                error=error_message,
            )

        except (OSError, IOError) as e:
            return ExecutionResult(success=False, error=str(e))
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _build_executable_code(
        self,
        code: str,
        test_code: str | None,
        entry_point: str | None,
    ) -> str:
        """
        Build executable code combining solution and test.

        Matches the generation phase logic in sessions.py for consistent behavior.
        """
        if not test_code:
            return f'{code}\n\nprint("TEST_PASSED")\n'

        # Deduplicate imports between code and test
        code_imports = set(re.findall(r"^(?:from|import)\s+.+$", code, re.MULTILINE))

        test_lines = []
        for line in test_code.split("\n"):
            if line.strip().startswith(("from ", "import ")):
                if line.strip() not in code_imports:
                    test_lines.append(line)
            else:
                test_lines.append(line)

        cleaned_test = "\n".join(test_lines)

        # Determine test execution trigger
        execution_trigger = self._get_test_execution_trigger(test_code, entry_point)

        return f"""{code}

{cleaned_test}{execution_trigger}

print("TEST_PASSED")
"""

    def _get_test_execution_trigger(
        self,
        test_code: str,
        entry_point: str | None,
    ) -> str:
        """
        Determine the test execution trigger.

        Matches the generation phase logic for invoking check() or test_* functions.
        """
        has_check = re.search(r"def\s+check\s*\(", test_code)
        has_test_func = re.search(r"def\s+(test_\w+)\s*\(", test_code)

        if has_check and entry_point:
            # Check if check() is already called with entry_point
            if re.search(rf"check\s*\(\s*{re.escape(entry_point)}\s*\)", test_code):
                return ""
            return f"\ncheck({entry_point})"
        elif has_test_func:
            test_name = has_test_func.group(1)
            return f"\n{test_name}()"

        return ""

    def _extract_error_message(self, stderr_text: str) -> str:
        """Extract meaningful error message from stderr.

        Parses Python tracebacks to extract:
        - The error type and message
        - The failing line of code (especially for AssertionError)
        - Import/attribute errors with full context
        """
        if not stderr_text:
            return "Unknown error"

        lines = stderr_text.split("\n")

        # Find the error line (last line starting with a known error type)
        error_line_idx = -1
        error_types = (
            "AssertionError",
            "TypeError",
            "ValueError",
            "AttributeError",
            "ImportError",
            "ModuleNotFoundError",
            "NameError",
            "KeyError",
            "IndexError",
            "RuntimeError",
            "SyntaxError",
            "IndentationError",
        )

        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line and (line.startswith(error_types) or "Error:" in line):
                error_line_idx = i
                break

        if error_line_idx == -1:
            return stderr_text[-500:].strip()

        error_line = lines[error_line_idx].strip()

        # For AssertionError, find the assertion line that failed
        if error_line.startswith("AssertionError"):
            return self._extract_assertion_context(lines, error_line_idx, error_line)

        # For AttributeError/ImportError, include full message
        if any(
            error_line.startswith(e)
            for e in ("AttributeError", "ImportError", "ModuleNotFoundError")
        ):
            return error_line

        return error_line if error_line else stderr_text[-500:].strip()

    def _extract_assertion_context(
        self,
        lines: list[str],
        error_line_idx: int,
        error_line: str,
    ) -> str:
        """Extract context for assertion errors."""
        # Look backwards for the assertion statement
        for i in range(error_line_idx - 1, max(0, error_line_idx - 10), -1):
            line = lines[i].strip()
            if line.startswith("assert "):
                if error_line == "AssertionError":
                    return f"AssertionError at: {line}"
                return f"{error_line} at: {line}"

        # If no assert found, look for the "File" line with context
        for i in range(error_line_idx - 1, max(0, error_line_idx - 5), -1):
            if "File " in lines[i] and ", line " in lines[i]:
                if i + 1 < error_line_idx:
                    code_line = lines[i + 1].strip()
                    if error_line == "AssertionError":
                        return f"AssertionError at: {code_line}"
                    return f"{error_line} at: {code_line}"
                break

        return error_line
