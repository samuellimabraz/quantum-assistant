"""Safe code execution sandbox for evaluating generated code."""

import multiprocessing
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Any


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    output: str = ""
    error: str = ""
    timeout: bool = False
    return_value: Any = None


class CodeExecutor:
    """Execute code safely in an isolated environment."""

    def __init__(self, timeout: int = 30):
        """
        Initialize code executor.

        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout

    def execute(
        self, code: str, test_code: str | None = None, entry_point: str | None = None
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
        # Use multiprocessing for timeout and isolation
        manager = multiprocessing.Manager()
        result_dict = manager.dict()

        process = multiprocessing.Process(
            target=self._run_code,
            args=(code, test_code, entry_point, result_dict),
        )

        process.start()
        process.join(timeout=self.timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            return ExecutionResult(success=False, error="Execution timeout", timeout=True)

        if "error" in result_dict:
            return ExecutionResult(success=False, error=result_dict["error"])

        return ExecutionResult(
            success=result_dict.get("success", False),
            output=result_dict.get("output", ""),
            return_value=result_dict.get("return_value"),
        )

    @staticmethod
    def _run_code(
        code: str, test_code: str | None, entry_point: str | None, result_dict: dict
    ) -> None:
        """
        Internal method to run code in subprocess.

        Args:
            code: Code to execute
            test_code: Optional test code
            entry_point: Optional entry point function
            result_dict: Shared dict for returning results
        """
        stdout = StringIO()
        stderr = StringIO()

        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                # Create execution namespace
                exec_globals = {}

                # Execute the main code
                exec(code, exec_globals)

                # If test code is provided, run it
                if test_code:
                    # Prepare test environment
                    if entry_point and entry_point in exec_globals:
                        exec_globals["candidate"] = exec_globals[entry_point]

                    # Execute test
                    exec(test_code, exec_globals)

                    # Check if test defined a check function
                    if "check" in exec_globals and callable(exec_globals["check"]):
                        if entry_point and entry_point in exec_globals:
                            exec_globals["check"](exec_globals[entry_point])

            result_dict["success"] = True
            result_dict["output"] = stdout.getvalue()

        except AssertionError as e:
            result_dict["success"] = False
            result_dict["error"] = f"Test assertion failed: {str(e)}\n{stderr.getvalue()}"

        except Exception as e:
            result_dict["success"] = False
            error_trace = traceback.format_exc()
            result_dict["error"] = (
                f"{type(e).__name__}: {str(e)}\n{error_trace}\n{stderr.getvalue()}"
            )

    def execute_function(
        self, code: str, function_name: str, test_cases: list[dict[str, Any]]
    ) -> tuple[bool, list[bool]]:
        """
        Execute a function with multiple test cases.

        Args:
            code: Code containing the function
            function_name: Name of the function to test
            test_cases: List of test case dicts with 'input' and 'expected' keys

        Returns:
            Tuple of (all_passed, list of individual test results)
        """
        results = []

        for test_case in test_cases:
            test_input = test_case.get("input", {})
            expected = test_case.get("expected")

            # Create test code
            test_code = f"""
result = {function_name}(**{test_input})
assert result == {repr(expected)}, f"Expected {repr(expected)}, got {{result}}"
"""

            execution_result = self.execute(code, test_code, function_name)
            results.append(execution_result.success)

        return all(results), results
