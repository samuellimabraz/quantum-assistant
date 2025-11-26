"""Unit test generation and validation for Qiskit code samples.

This module provides test generation following Qiskit HumanEval patterns:
- Tests are generated alongside questions for code types
- Tests verify functional correctness, not just syntax
- Uses Qiskit's comparison utilities (Operator.equiv, Statevector.equiv)
"""

import asyncio
import ast
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from synthetic_data.models import LLMClient, Message


@dataclass
class TestResult:
    """Result of test execution."""

    passed: bool
    error_message: Optional[str] = None
    output: str = ""


@dataclass
class GeneratedTest:
    """Generated unit test with metadata."""

    test_code: str
    entry_point: str
    is_valid: bool = True
    validation_error: Optional[str] = None


class TestGenerator:
    """Generate and validate unit tests for Qiskit code samples.

    This generator creates pytest-compatible tests that verify code behavior
    using Qiskit's testing utilities for quantum circuit comparison.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        timeout_seconds: int = 60,
    ):
        """Initialize test generator.

        Args:
            llm_client: LLM client for test generation
            timeout_seconds: Timeout for test execution
        """
        self.llm_client = llm_client
        self.timeout_seconds = timeout_seconds

    async def generate_test_for_task_async(
        self,
        task_description: str,
        reference_code: str,
        entry_point: Optional[str] = None,
    ) -> GeneratedTest:
        """Generate a unit test for a code task.

        Args:
            task_description: What the code should do
            reference_code: The reference solution
            entry_point: Function name (extracted if not provided)

        Returns:
            GeneratedTest with test code and metadata
        """
        if not entry_point:
            entry_point = self._extract_entry_point(reference_code)

        if not entry_point:
            return GeneratedTest(
                test_code="",
                entry_point="",
                is_valid=False,
                validation_error="Could not identify function entry point",
            )

        system_prompt = self._get_test_generation_system_prompt()
        user_prompt = self._get_test_generation_prompt(
            task_description, reference_code, entry_point
        )

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        try:
            response = await self.llm_client.generate_async(messages, temperature=0.2)
            test_code = self._extract_test_code(response)

            if not test_code:
                return GeneratedTest(
                    test_code="",
                    entry_point=entry_point,
                    is_valid=False,
                    validation_error="Could not extract test code from response",
                )

            # Validate test structure
            validation_error = self._validate_test_structure(test_code, entry_point)
            if validation_error:
                return GeneratedTest(
                    test_code=test_code,
                    entry_point=entry_point,
                    is_valid=False,
                    validation_error=validation_error,
                )

            return GeneratedTest(
                test_code=test_code,
                entry_point=entry_point,
                is_valid=True,
            )

        except Exception as e:
            return GeneratedTest(
                test_code="",
                entry_point=entry_point,
                is_valid=False,
                validation_error=f"Test generation failed: {str(e)}",
            )

    async def validate_code_against_test_async(
        self,
        code: str,
        test_code: str,
        entry_point: str,
        prompt: str = "",
        question_type: str = "",
    ) -> TestResult:
        """Validate code by running the unit test.

        Args:
            code: Solution code to test
            test_code: Unit test code
            entry_point: Function name being tested
            prompt: Original prompt (needed for function_completion)
            question_type: Type of question (function_completion or code_generation)

        Returns:
            TestResult indicating pass/fail and any errors
        """
        # Combine code and test
        combined_code = self._combine_code_and_test(
            code, test_code, entry_point, prompt, question_type
        )

        # Run in subprocess
        return await self._execute_test_async(combined_code)

    async def generate_batch_tests_async(
        self,
        tasks: list[dict],
        max_concurrent: int = 5,
        progress_callback=None,
    ) -> list[GeneratedTest]:
        """Generate tests for multiple tasks concurrently.

        Args:
            tasks: List of dicts with 'task_description', 'reference_code', 'entry_point'
            max_concurrent: Maximum concurrent requests
            progress_callback: Optional callback(completed_count)

        Returns:
            List of GeneratedTest results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        completed = [0]
        lock = asyncio.Lock()

        async def generate_single(task: dict) -> GeneratedTest:
            async with semaphore:
                result = await self.generate_test_for_task_async(
                    task["task_description"],
                    task["reference_code"],
                    task.get("entry_point"),
                )
                async with lock:
                    completed[0] += 1
                    if progress_callback:
                        progress_callback(completed[0])
                return result

        results = await asyncio.gather(*[generate_single(t) for t in tasks])
        return list(results)

    def _get_test_generation_system_prompt(self) -> str:
        """Get system prompt for test generation."""
        return """You are an expert at writing unit tests for Qiskit quantum computing code.
Generate complete pytest unit tests following Qiskit HumanEval patterns.

QISKIT TESTING UTILITIES:
- Operator.equiv() for circuit/operator comparison
- Statevector.equiv() for state comparison  
- numpy.allclose() for numerical comparisons
- Statevector.from_instruction(circuit) for statevector extraction
- Operator(circuit) for unitary extraction

TEST FORMAT (Qiskit HumanEval style):
```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
# other imports as needed

def check(candidate):
    result = candidate()  # or candidate(args)
    # assertions here
    assert result.equiv(expected)  # for quantum objects
    assert isinstance(result, ExpectedType)

check(entry_point_name)
```

REQUIREMENTS:
- Use check(candidate) pattern from Qiskit HumanEval
- Include ALL necessary imports
- NO comments in test code
- Tests must be deterministic and reproducible
- For circuits with measurements, check counts distribution patterns
- Test must verify FUNCTION behavior, not implementation details

EXAMPLE TESTS:
```python
# Statevector test
from qiskit.quantum_info import Statevector
from math import sqrt
def check(candidate):
    result = candidate()
    solution = (Statevector.from_label("00") + Statevector.from_label("11")) / sqrt(2)
    assert result.equiv(solution)
check(create_bell_statevector)

# Circuit operator test
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
def check(candidate):
    result = candidate(3)
    assert isinstance(result, QuantumCircuit)
    assert result.num_qubits == 3
check(create_quantum_circuit)

# Measurement counts test
def check(candidate):
    result = candidate()
    assert isinstance(result, dict)
    assert result.keys() == {"00", "11"}
    assert 0.4 < (result["00"] / sum(result.values())) < 0.6
check(run_bell_state_simulator)
```

Return ONLY a single Python code block with the complete test."""

    def _get_test_generation_prompt(
        self, task_description: str, reference_code: str, entry_point: str
    ) -> str:
        """Get user prompt for test generation."""
        return f"""Generate a Qiskit HumanEval style unit test for this code.

Task: {task_description}

Reference Solution:
```python
{reference_code}
```

Entry Point: {entry_point}

Create a test using the check(candidate) pattern:
```python
def check(candidate):
    result = candidate()  # call with appropriate args
    # verify result using assertions
    # use .equiv() for quantum objects

check({entry_point})
```

IMPORTANT:
- Include ALL necessary imports at the top
- Call {entry_point} with correct arguments
- Use Operator.equiv() for circuit comparison
- Use Statevector.equiv() for state comparison
- For measurements, check distribution patterns
- NO comments in code

Return ONLY the complete test code."""

    def _extract_entry_point(self, code: str) -> Optional[str]:
        """Extract the main function name from code."""
        # Find all function definitions
        matches = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
        if not matches:
            return None

        # Return the first non-private function, or first function
        for match in matches:
            if not match.startswith("_"):
                return match
        return matches[0]

    def _extract_test_code(self, response: str) -> str:
        """Extract test code from LLM response."""
        # Try to extract from code blocks
        code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)

        if code_blocks:
            # Return the longest code block (likely the complete test)
            return max(code_blocks, key=len).strip()

        # If no code blocks, return stripped response
        return response.strip()

    def _validate_test_structure(self, test_code: str, entry_point: str) -> Optional[str]:
        """Validate test code structure. Returns error message or None if valid."""
        # Check syntax
        try:
            ast.parse(test_code)
        except SyntaxError as e:
            return f"Syntax error in test: {e.msg}"

        # Check for test function (either check(candidate) or test_ pattern)
        has_check = re.search(r"def\s+check\s*\(", test_code)
        has_test = re.search(r"def\s+test_", test_code)
        if not has_check and not has_test:
            return "Test must contain a check(candidate) or test_ function"

        # Check for assertions
        if not re.search(r"assert\s+", test_code):
            return "Test must contain assertions"

        # Check that entry point is called
        if entry_point not in test_code:
            return f"Test must call the entry point function '{entry_point}'"

        return None

    def _combine_code_and_test(
        self,
        code: str,
        test_code: str,
        entry_point: str,
        prompt: str = "",
        question_type: str = "",
    ) -> str:
        """Combine solution code and test code for execution.
        
        For function_completion: prompt contains stub, code contains body only.
        For code_generation: code contains full solution.
        
        Args:
            code: The solution code
            test_code: The unit test code
            entry_point: Function name being tested
            prompt: Original prompt (needed for function_completion to get stub)
            question_type: Type of question (function_completion or code_generation)
        """
        # Determine if this is function_completion format (prompt has stub)
        is_function_completion = (
            question_type == "function_completion" or
            (prompt and "def " in prompt and "pass" in prompt and "def " not in code)
        )
        
        if is_function_completion and prompt:
            # Combine stub from prompt with body from answer
            full_code = self._assemble_function_completion(prompt, code, entry_point)
        else:
            # code_generation - answer already has full code
            full_code = code
        
        # Remove duplicate imports from test if they exist in code
        code_imports = set(re.findall(r"^(?:from|import)\s+.+$", full_code, re.MULTILINE))

        test_lines = []
        for line in test_code.split("\n"):
            # Skip import lines that already exist in code
            if line.strip().startswith(("from ", "import ")):
                if line.strip() not in code_imports:
                    test_lines.append(line)
            else:
                test_lines.append(line)

        cleaned_test = "\n".join(test_lines)

        # Determine test execution pattern
        has_check = re.search(r"def\s+check\s*\(", test_code)
        has_test_func = re.search(r"def\s+(test_\w+)\s*\(", test_code)

        if has_check:
            # Qiskit HumanEval pattern - check if check() is already called
            if re.search(rf"check\s*\(\s*{re.escape(entry_point)}\s*\)", test_code):
                execution = ""
            else:
                execution = f"\ncheck({entry_point})"
        elif has_test_func:
            test_name = has_test_func.group(1)
            execution = f"\n{test_name}()"
        else:
            execution = ""

        combined = f"""{full_code}

{cleaned_test}{execution}

print("TEST_PASSED")
"""
        return combined

    def _assemble_function_completion(
        self, prompt: str, body: str, entry_point: str
    ) -> str:
        """Assemble function completion from stub (prompt) and body (answer).
        
        The prompt contains: imports + def signature + docstring + pass
        The body contains: the actual implementation
        
        Args:
            prompt: The function stub with imports, signature, docstring, pass
            body: The function body (implementation)
            entry_point: Function name
            
        Returns:
            Complete executable code
        """
        # Extract code block from prompt if wrapped in markdown
        code_match = re.search(r"```(?:python)?\s*\n(.*?)```", prompt, re.DOTALL)
        stub = code_match.group(1).strip() if code_match else prompt.strip()
        
        # Clean up body - remove code block markers if present
        code_match = re.search(r"```(?:python)?\s*\n(.*?)```", body, re.DOTALL)
        clean_body = code_match.group(1).strip() if code_match else body.strip()
        
        # Find the 'pass' statement and replace it with the body
        # Handle indentation - the pass is usually indented
        lines = stub.split("\n")
        result_lines = []
        replaced = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "pass" and not replaced:
                # Get indentation from the pass line
                indent = line[:len(line) - len(line.lstrip())]
                if not indent:
                    indent = "    "  # Default to 4 spaces
                
                # Add indented body
                for body_line in clean_body.split("\n"):
                    if body_line.strip():
                        result_lines.append(indent + body_line.strip())
                    else:
                        result_lines.append("")
                replaced = True
            else:
                result_lines.append(line)
        
        # If pass wasn't found, append body after function definition
        if not replaced:
            result_lines.append("")
            for body_line in clean_body.split("\n"):
                result_lines.append("    " + body_line.strip() if body_line.strip() else "")
        
        return "\n".join(result_lines)

    async def _execute_test_async(self, combined_code: str) -> TestResult:
        """Execute combined code and test in subprocess."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(combined_code)
            temp_path = f.name

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return TestResult(
                    passed=False,
                    error_message=f"Test timeout (>{self.timeout_seconds}s)",
                )

            stdout_text = stdout.decode("utf-8").strip()
            stderr_text = stderr.decode("utf-8").strip()

            if process.returncode == 0 and "TEST_PASSED" in stdout_text:
                return TestResult(passed=True, output=stdout_text)
            else:
                # Extract relevant error
                error_lines = stderr_text.split("\n")
                relevant_error = next(
                    (
                        line
                        for line in reversed(error_lines)
                        if "Error:" in line or "assert" in line.lower()
                    ),
                    stderr_text[-500:] if stderr_text else "Unknown error",
                )
                return TestResult(
                    passed=False,
                    error_message=relevant_error,
                    output=stdout_text,
                )

        except (OSError, IOError) as e:
            return TestResult(passed=False, error_message=str(e))
        finally:
            Path(temp_path).unlink(missing_ok=True)


class CodeWithTestValidator:
    """Validate code samples against their unit tests.

    Used during sample generation to ensure answer code passes the test.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_correction_attempts: int = 3,
        timeout_seconds: int = 60,
    ):
        """Initialize validator.

        Args:
            llm_client: LLM client for code correction
            max_correction_attempts: Max attempts to fix failing code
            timeout_seconds: Timeout for test execution
        """
        self.llm_client = llm_client
        self.test_generator = TestGenerator(llm_client, timeout_seconds)
        self.max_attempts = max_correction_attempts

    async def validate_and_correct_async(
        self,
        code: str,
        test_code: str,
        entry_point: str,
        task_description: str = "",
        question_type: str = "",
    ) -> tuple[bool, str, int]:
        """Validate code against test and attempt correction if needed.

        Args:
            code: Solution code to validate
            test_code: Unit test code
            entry_point: Function name being tested
            task_description: Original task/prompt (for correction context and assembly)
            question_type: Type of question (function_completion or code_generation)

        Returns:
            Tuple of (passed, corrected_code, attempts_used)
        """
        current_code = code

        for attempt in range(self.max_attempts):
            result = await self.test_generator.validate_code_against_test_async(
                current_code, test_code, entry_point, task_description, question_type
            )

            if result.passed:
                return True, current_code, attempt + 1

            # Attempt correction
            corrected = await self._request_correction_async(
                current_code,
                test_code,
                entry_point,
                result.error_message,
                task_description,
                question_type,
            )

            if not corrected or corrected == current_code:
                break

            current_code = corrected

        return False, current_code, self.max_attempts

    async def _request_correction_async(
        self,
        code: str,
        test_code: str,
        entry_point: str,
        error_message: str,
        task_description: str,
        question_type: str = "",
    ) -> Optional[str]:
        """Request code correction from LLM."""
        is_function_completion = question_type == "function_completion"
        
        if is_function_completion:
            system_prompt = """You are a Qiskit code correction expert.
Fix the function body to pass the unit test.
Return ONLY the corrected function body (the code that goes inside the function).
Do NOT include imports or the function signature - only the body.

Requirements:
- Return only the function body
- Fix the specific error indicated
- Maintain Qiskit best practices"""
            
            user_prompt = f"""The function body fails its unit test. Fix it.

Function Stub (prompt):
{task_description}

Current Body (your response should be similar format):
{code}

Unit Test:
```python
{test_code}
```

Error: {error_message}

Provide ONLY the corrected function body."""
        else:
            system_prompt = """You are a Qiskit code correction expert.
Fix the code to pass the unit test while maintaining correct quantum computing logic.
Return ONLY the corrected Python code in a single code block.

Requirements:
- Include ALL necessary imports
- Keep the same function signature
- Fix the specific error indicated
- Maintain Qiskit best practices"""

            user_prompt = f"""The following code fails its unit test. Fix it.

Task: {task_description}

Current Code:
```python
{code}
```

Unit Test:
```python
{test_code}
```

Error: {error_message}

Provide the corrected code that will pass the test."""

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        try:
            response = await self.llm_client.generate_async(messages, temperature=0.1)
            code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)

            if code_blocks:
                return code_blocks[0].strip()

            # Fallback if no code block markers
            if "import" in response or "def " in response or "return" in response:
                return response.strip()

            return None

        except Exception:
            return None
