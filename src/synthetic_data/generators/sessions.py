"""Answer generation sessions with validation loop."""

import asyncio
import ast
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from synthetic_data.config import QuestionType
from synthetic_data.generators.types import InputCandidate
from synthetic_data.models import LLMClient, Message
from synthetic_data.utils.tracer import GenerationTracer, ConversationTrace


IBM_SERVICE_ERROR_PATTERNS = [
    "invalidaccounterror",
    "accountalreadyexistserror",
    "qiskitserverlessexception",
    "qiskit_ibm_token",
    "ibm quantum api token",
    "unable to retrieve instances",
    "_verify_credentials",
]

DEPRECATED_API_PATTERNS = [
    (r"qubits\[\d*\]\.index\b", "Qubit.index removed - use circuit.find_bit(qubit).index"),
    (r"clbits\[\d*\]\.index\b", "Clbit.index removed - use circuit.find_bit(clbit).index"),
    (r"\bq\.index\b(?!\.)", "Qubit.index removed - use circuit.find_bit(q).index"),
    (r"\bqb\.index\b(?!\.)", "Qubit.index removed - use circuit.find_bit(qb).index"),
    (r"\.c_if\s*\(", "c_if removed - use QuantumCircuit.if_test() context manager"),
    (r"\.quasi_dists\b", "quasi_dists removed - use result.data for sampler results"),
    (r"\.true_body\b", "IfElseOp.true_body removed - use .blocks[0]"),
    (r"\.false_body\b", "IfElseOp.false_body removed - use .blocks[1]"),
    (r"from\s+qiskit\.opflow", "qiskit.opflow removed - use qiskit.quantum_info"),
    (r"from\s+qiskit\.pulse", "qiskit.pulse removed in Qiskit 2.0"),
    (r"from\s+qiskit\s+import\s+.*\bpulse\b", "qiskit.pulse removed in Qiskit 2.0"),
    (r"\.bind_parameters\s*\(", "bind_parameters deprecated - use assign_parameters"),
    (r"qiskit\.execute\s*\(", "qiskit.execute removed - use Sampler or Estimator"),
    (r"\.qasm\s*\(", "qasm() removed - use qasm2.dumps() or qasm3.dumps()"),
]


def _check_deprecated_apis(code: str) -> Optional[tuple[str, str]]:
    if not code:
        return None
    for pattern, message in DEPRECATED_API_PATTERNS:
        match = re.search(pattern, code)
        if match:
            return (pattern, message)
    return None


def _get_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["MallocStackLogging"] = "0"
    env["MallocNanoZone"] = "0"
    return env


def _is_ibm_service_error(error_message: str) -> bool:
    if not error_message:
        return False
    error_lower = error_message.lower()
    return any(pattern in error_lower for pattern in IBM_SERVICE_ERROR_PATTERNS)


@dataclass
class AnswerResult:
    """Result from answer generation with validation."""

    answer: str
    passed: bool = False
    iterations_used: int = 0
    error_history: list[dict] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of code validation."""

    passed: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class AnswerSession:
    """Session for generating and validating answers.

    Maintains conversation context for the correction loop,
    allowing the model to see its previous attempts and errors.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        answer_prompt: str,
        correction_prompt: str,
        candidate: InputCandidate,
        max_iterations: int = 3,
        timeout_seconds: int = 60,
        tracer: Optional[GenerationTracer] = None,
        skip_ibm_service_validation: bool = True,
    ):
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.answer_prompt = answer_prompt
        self.correction_prompt = correction_prompt
        self.candidate = candidate
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.tracer = tracer
        self.skip_ibm_service_validation = skip_ibm_service_validation
        self.messages: list[Message] = []
        self._conversation: Optional[ConversationTrace] = None

    async def generate_and_validate_async(self) -> AnswerResult:
        """Generate answer with validation loop.

        Returns:
            AnswerResult with final answer and validation status
        """
        # Start conversation trace
        if self.tracer:
            self._conversation = self.tracer.start_conversation(
                stage="answer_generation",
                question_type=self.candidate.question_type.value,
                entry_point=self.candidate.entry_point,
                is_multimodal=self.candidate.is_multimodal,
                has_test=bool(self.candidate.test_code),
            )
            self.tracer.log_message(self._conversation, "system", self.system_prompt, iteration=0)
            self.tracer.log_message(
                self._conversation, "user", self.answer_prompt, iteration=0, step="answer_request"
            )

        self.messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=self.answer_prompt),
        ]

        error_history = []

        try:
            answer = await self.llm_client.generate_async(self.messages)

            if self.tracer and self._conversation:
                self.tracer.log_message(
                    self._conversation, "assistant", answer, iteration=1, step="initial_answer"
                )

            answer = self._clean_answer(answer)

            if not answer:
                if self.tracer and self._conversation:
                    self.tracer.complete_conversation(
                        self._conversation, success=False, error="Empty answer"
                    )
                return AnswerResult(
                    answer="",
                    passed=False,
                    error_history=[{"iteration": 0, "error": "Empty answer"}],
                )

            self.messages.append(Message(role="assistant", content=answer))

        except Exception as e:
            if self.tracer and self._conversation:
                self.tracer.complete_conversation(self._conversation, success=False, error=str(e))
            return AnswerResult(
                answer="",
                passed=False,
                error_history=[{"iteration": 0, "error": str(e)}],
            )

        # QA type: basic code verification if code is present
        if self.candidate.question_type == QuestionType.QA:
            return await self._verify_qa_answer_async(answer, error_history)

        # Code types: run test validation loop
        return await self._validate_code_answer_async(answer, error_history)

    async def _verify_qa_answer_async(self, answer: str, error_history: list[dict]) -> AnswerResult:
        """Verify QA answer code blocks for syntax/execution."""
        code_blocks = self._extract_code_blocks(answer)

        if not code_blocks:
            if self.tracer and self._conversation:
                self.tracer.complete_conversation(
                    self._conversation,
                    success=True,
                    total_iterations=1,
                    note="QA without code - auto-pass",
                )
            return AnswerResult(answer=answer, passed=True)

        main_code = max(code_blocks, key=len)
        current_answer = answer

        if self.tracer and self._conversation:
            self.tracer.log_entry(
                stage="answer_generation",
                step_type="qa_code_verification",
                content=f"Found {len(code_blocks)} code blocks, verifying main block ({len(main_code)} chars)",
                candidate_id=self._conversation.id,
                iteration=0,
            )

        for iteration in range(self.max_iterations):
            validation = await self._validate_code_async(main_code)

            if self.tracer and self._conversation:
                self.tracer.log_entry(
                    stage="answer_generation",
                    step_type="qa_validation",
                    content=f"Passed: {validation.passed}, Error: {validation.error_message or 'None'}",
                    candidate_id=self._conversation.id,
                    iteration=iteration + 1,
                )

            if validation.passed:
                if self.tracer and self._conversation:
                    self.tracer.complete_conversation(
                        self._conversation,
                        success=True,
                        total_iterations=iteration + 1,
                    )
                return AnswerResult(
                    answer=current_answer,
                    passed=True,
                    iterations_used=iteration + 1,
                    error_history=error_history,
                )

            error_history.append(
                {
                    "iteration": iteration + 1,
                    "error_type": validation.error_type,
                    "error": validation.error_message,
                }
            )

            correction_msg = self.correction_prompt.format(
                error_type=validation.error_type,
                error_message=validation.error_message,
            )
            self.messages.append(Message(role="user", content=correction_msg))

            if self.tracer and self._conversation:
                self.tracer.log_message(
                    self._conversation,
                    "user",
                    correction_msg,
                    iteration=iteration + 2,
                    step="qa_correction_request",
                )

            try:
                corrected = await self.llm_client.generate_async(self.messages, temperature=1.0)

                if self.tracer and self._conversation:
                    self.tracer.log_message(
                        self._conversation,
                        "assistant",
                        corrected,
                        iteration=iteration + 2,
                        step="qa_correction_response",
                    )

                corrected_code = self._extract_code_blocks(corrected)

                if corrected_code:
                    main_code = max(corrected_code, key=len)
                    current_answer = self._replace_code_in_answer(current_answer, main_code)
                    self.messages.append(Message(role="assistant", content=corrected))
                else:
                    break
            except Exception as e:
                if self.tracer and self._conversation:
                    self.tracer.log_entry(
                        stage="answer_generation",
                        step_type="qa_correction_error",
                        content=str(e),
                        candidate_id=self._conversation.id,
                        iteration=iteration + 2,
                    )
                break

        if self.tracer and self._conversation:
            self.tracer.complete_conversation(
                self._conversation,
                success=False,
                total_iterations=self.max_iterations,
                error_history=error_history,
            )

        return AnswerResult(
            answer=current_answer,
            passed=False,
            iterations_used=self.max_iterations,
            error_history=error_history,
        )

    async def _validate_code_answer_async(
        self, answer: str, error_history: list[dict]
    ) -> AnswerResult:
        """Validate code answer against unit test with correction loop."""
        if not self.candidate.test_code or not self.candidate.entry_point:
            if self.tracer and self._conversation:
                self.tracer.complete_conversation(
                    self._conversation,
                    success=True,
                    total_iterations=1,
                    note="No test code - auto-pass",
                )
            return AnswerResult(answer=answer, passed=True)

        current_answer = answer

        if self.tracer and self._conversation:
            self.tracer.log_entry(
                stage="answer_generation",
                step_type="test_code",
                content=self.candidate.test_code,
                candidate_id=self._conversation.id,
                iteration=0,
            )

        for iteration in range(self.max_iterations):
            deprecated = _check_deprecated_apis(current_answer)
            if deprecated:
                _, message = deprecated
                test_result = ValidationResult(
                    passed=False,
                    error_type="deprecated_api",
                    error_message=f"Code uses deprecated Qiskit 2.0 API: {message}",
                )
            else:
                test_result = await self._run_test_async(current_answer)

            if self.tracer and self._conversation:
                self.tracer.log_entry(
                    stage="answer_generation",
                    step_type="test_validation",
                    content=f"Passed: {test_result.passed}, Error: {test_result.error_message or 'None'}",
                    candidate_id=self._conversation.id,
                    iteration=iteration + 1,
                    error_type=test_result.error_type,
                )

            if test_result.passed:
                if self.tracer and self._conversation:
                    self.tracer.complete_conversation(
                        self._conversation,
                        success=True,
                        total_iterations=iteration + 1,
                        final_answer=current_answer,
                    )
                return AnswerResult(
                    answer=current_answer,
                    passed=True,
                    iterations_used=iteration + 1,
                    error_history=error_history,
                )

            if self.skip_ibm_service_validation and _is_ibm_service_error(
                test_result.error_message
            ):
                if self.tracer and self._conversation:
                    self.tracer.complete_conversation(
                        self._conversation,
                        success=True,
                        total_iterations=iteration + 1,
                        final_answer=current_answer,
                        note="IBM service error skipped",
                    )
                return AnswerResult(
                    answer=current_answer,
                    passed=True,
                    iterations_used=iteration + 1,
                    error_history=error_history,
                )

            error_history.append(
                {
                    "iteration": iteration + 1,
                    "error_type": test_result.error_type or "test_failure",
                    "error": test_result.error_message,
                }
            )

            correction_msg = self.correction_prompt.format(
                error_type=test_result.error_type or "test",
                error_message=test_result.error_message,
            )
            self.messages.append(Message(role="user", content=correction_msg))

            # Log correction request
            if self.tracer and self._conversation:
                self.tracer.log_message(
                    self._conversation,
                    "user",
                    correction_msg,
                    iteration=iteration + 2,
                    step="correction_request",
                    error_type=test_result.error_type,
                    error_message=test_result.error_message,
                )

            try:
                corrected = await self.llm_client.generate_async(self.messages, temperature=1.0)

                # Log correction response
                if self.tracer and self._conversation:
                    self.tracer.log_message(
                        self._conversation,
                        "assistant",
                        corrected,
                        iteration=iteration + 2,
                        step="correction_response",
                    )

                corrected = self._clean_answer(corrected)

                if corrected and corrected != current_answer:
                    current_answer = corrected
                    self.messages.append(Message(role="assistant", content=corrected))
                else:
                    if self.tracer and self._conversation:
                        self.tracer.log_entry(
                            stage="answer_generation",
                            step_type="correction_failed",
                            content="No meaningful change in correction",
                            candidate_id=self._conversation.id,
                            iteration=iteration + 2,
                        )
                    break
            except Exception as e:
                if self.tracer and self._conversation:
                    self.tracer.log_entry(
                        stage="answer_generation",
                        step_type="correction_error",
                        content=str(e),
                        candidate_id=self._conversation.id,
                        iteration=iteration + 2,
                    )
                break

        if self.tracer and self._conversation:
            self.tracer.complete_conversation(
                self._conversation,
                success=False,
                total_iterations=self.max_iterations,
                final_answer=current_answer,
                error_history=error_history,
            )

        return AnswerResult(
            answer=current_answer,
            passed=False,
            iterations_used=self.max_iterations,
            error_history=error_history,
        )

    async def _run_test_async(self, answer: str) -> ValidationResult:
        """Run unit test against answer code."""
        combined_code = self._combine_code_and_test(answer)

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
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ValidationResult(
                    passed=False,
                    error_type="timeout",
                    error_message=f"Timeout (>{self.timeout_seconds}s)",
                )

            stdout_text = stdout.decode("utf-8").strip()
            stderr_text = stderr.decode("utf-8").strip()

            if process.returncode == 0 and "TEST_PASSED" in stdout_text:
                return ValidationResult(passed=True)

            error_message = self._extract_error_message(stderr_text)
            return ValidationResult(
                passed=False,
                error_type="test_failure",
                error_message=error_message,
            )

        except (OSError, IOError) as e:
            return ValidationResult(
                passed=False,
                error_type="execution",
                error_message=str(e),
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)

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
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line and (
                line.startswith(
                    (
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
                )
                or "Error:" in line
            ):
                error_line_idx = i
                break

        if error_line_idx == -1:
            # No standard error found, return last 300 chars
            return stderr_text[-300:].strip()

        error_line = lines[error_line_idx].strip()

        # For AssertionError, find the assertion line that failed
        if error_line.startswith("AssertionError"):
            # Look backwards for the assertion statement
            for i in range(error_line_idx - 1, max(0, error_line_idx - 10), -1):
                line = lines[i].strip()
                if line.startswith("assert "):
                    # Found the assertion - include it with the error
                    if error_line == "AssertionError":
                        return f"AssertionError at: {line}"
                    else:
                        return f"{error_line} at: {line}"

            # If no assert found, look for the "File" line with context
            for i in range(error_line_idx - 1, max(0, error_line_idx - 5), -1):
                if "File " in lines[i] and ", line " in lines[i]:
                    # Include the code line after the File line
                    if i + 1 < error_line_idx:
                        code_line = lines[i + 1].strip()
                        if error_line == "AssertionError":
                            return f"AssertionError at: {code_line}"
                        else:
                            return f"{error_line} at: {code_line}"
                    break

        # For AttributeError/ImportError, include full message
        if any(
            error_line.startswith(e)
            for e in ("AttributeError", "ImportError", "ModuleNotFoundError")
        ):
            return error_line

        # For other errors, check if there's useful context
        if ":" in error_line:
            return error_line

        return error_line if error_line else stderr_text[-300:].strip()

    async def _validate_code_async(self, code: str) -> ValidationResult:
        """Validate code syntax and execution."""
        try:
            ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(
                passed=False,
                error_type="syntax",
                error_message=f"Syntax error at line {e.lineno}: {e.msg}",
            )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
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
                _, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ValidationResult(
                    passed=False,
                    error_type="timeout",
                    error_message=f"Timeout (>{self.timeout_seconds}s)",
                )

            if process.returncode != 0:
                stderr_text = stderr.decode("utf-8").strip()
                error_lines = stderr_text.split("\n")
                relevant_error = next(
                    (line for line in reversed(error_lines) if "Error:" in line),
                    stderr_text,
                )
                return ValidationResult(
                    passed=False,
                    error_type="execution",
                    error_message=f"Execution error: {relevant_error}",
                )

            return ValidationResult(passed=True)

        except (OSError, IOError) as e:
            return ValidationResult(
                passed=False,
                error_type="execution",
                error_message=str(e),
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _combine_code_and_test(self, answer: str) -> str:
        """Combine answer code with unit test."""
        question_type = self.candidate.question_type
        question = self.candidate.question

        if question_type == QuestionType.FUNCTION_COMPLETION and question:
            full_code = self._assemble_function_completion(answer)
        else:
            full_code = answer

        # Remove duplicate imports
        code_imports = set(re.findall(r"^(?:from|import)\s+.+$", full_code, re.MULTILINE))

        test_lines = []
        for line in self.candidate.test_code.split("\n"):
            if line.strip().startswith(("from ", "import ")):
                if line.strip() not in code_imports:
                    test_lines.append(line)
            else:
                test_lines.append(line)

        cleaned_test = "\n".join(test_lines)

        # Determine test execution
        has_check = re.search(r"def\s+check\s*\(", self.candidate.test_code)
        has_test_func = re.search(r"def\s+(test_\w+)\s*\(", self.candidate.test_code)

        if has_check:
            entry_point = self.candidate.entry_point
            if re.search(rf"check\s*\(\s*{re.escape(entry_point)}\s*\)", self.candidate.test_code):
                execution = ""
            else:
                execution = f"\ncheck({entry_point})"
        elif has_test_func:
            test_name = has_test_func.group(1)
            execution = f"\n{test_name}()"
        else:
            execution = ""

        return f"""{full_code}

{cleaned_test}{execution}

print("TEST_PASSED")
"""

    def _assemble_function_completion(self, body: str) -> str:
        """Assemble function completion from stub and body.

        Handles the common case where the model returns body code where:
        - First line has 0 indentation
        - Subsequent lines have indentation relative to the first line

        This normalizes all body lines to the target indentation level.
        """
        question = self.candidate.question

        # Extract stub from question (may be wrapped in markdown)
        code_match = re.search(r"```(?:python)?\s*\n(.*?)```", question, re.DOTALL)
        stub = code_match.group(1) if code_match else question
        stub = stub.rstrip()

        # Extract body from answer (may be wrapped in markdown)
        code_match = re.search(r"```(?:python)?\s*\n(.*?)```", body, re.DOTALL)
        clean_body = code_match.group(1) if code_match else body
        clean_body = clean_body.rstrip()

        # Normalize body indentation
        body_lines = clean_body.split("\n")
        normalized_body = self._normalize_body_indentation(body_lines)

        # Find and replace the pass statement
        stub_lines = stub.split("\n")
        result_lines = []
        replaced = False

        for line in stub_lines:
            stripped = line.strip()
            if stripped == "pass" and not replaced:
                # Found the pass statement - get its indentation
                pass_indent = len(line) - len(line.lstrip())
                base_indent = " " * pass_indent

                # Insert normalized body lines with target indentation
                for body_line in normalized_body:
                    if body_line.strip():
                        # Body line already has relative indentation, add base
                        result_lines.append(base_indent + body_line)
                    else:
                        result_lines.append("")
                replaced = True
            else:
                result_lines.append(line)

        if not replaced:
            # No pass found - append body with default 4-space indent
            result_lines.append("")
            for body_line in normalized_body:
                if body_line.strip():
                    result_lines.append("    " + body_line)
                else:
                    result_lines.append("")

        return "\n".join(result_lines)

    def _normalize_body_indentation(self, body_lines: list[str]) -> list[str]:
        """Normalize body indentation to be consistent.

        Handles cases where model output has inconsistent indentation patterns.
        """
        non_empty_lines = [(i, line) for i, line in enumerate(body_lines) if line.strip()]

        if not non_empty_lines:
            return body_lines

        _, first_line = non_empty_lines[0]
        first_indent = len(first_line) - len(first_line.lstrip())
        first_stripped = first_line.strip()

        if first_indent > 0 or len(non_empty_lines) < 2:
            return self._standard_normalize(body_lines, non_empty_lines)

        subsequent_indents = [len(line) - len(line.lstrip()) for _, line in non_empty_lines[1:]]
        if not subsequent_indents:
            return body_lines

        is_block_start = first_stripped.endswith(":")

        if not is_block_start:
            min_subsequent = min(subsequent_indents)
            if min_subsequent > 0:
                return self._normalize_shifted_block(body_lines, non_empty_lines)
            return self._standard_normalize(body_lines, non_empty_lines)

        unique_indents = sorted(set(subsequent_indents))
        if len(unique_indents) < 2:
            return self._standard_normalize(body_lines, non_empty_lines)

        min_indent = unique_indents[0]
        second_min = unique_indents[1] if len(unique_indents) > 1 else min_indent

        if min_indent == 4 and second_min == 8:
            return self._normalize_shifted_block(body_lines, non_empty_lines)

        continuation_keywords = ("elif ", "else:", "except ", "except:", "finally:")
        has_continuation = any(
            any(line.strip().startswith(kw) for kw in continuation_keywords)
            for _, line in non_empty_lines[1:]
        )

        if has_continuation:
            return self._normalize_with_continuations(body_lines, non_empty_lines)

        return self._standard_normalize(body_lines, non_empty_lines)

    def _standard_normalize(self, body_lines: list[str], non_empty_lines: list[tuple]) -> list[str]:
        """Standard normalization: subtract minimum indent from all lines."""
        all_indents = [len(line) - len(line.lstrip()) for _, line in non_empty_lines]
        min_indent = min(all_indents) if all_indents else 0

        result = []
        for line in body_lines:
            if not line.strip():
                result.append("")
            else:
                current_indent = len(line) - len(line.lstrip())
                relative = current_indent - min_indent
                result.append(" " * relative + line.lstrip())

        return result

    def _normalize_shifted_block(
        self, body_lines: list[str], non_empty_lines: list[tuple]
    ) -> list[str]:
        """Normalize blocks where code is shifted by 4 extra spaces."""
        first_idx = non_empty_lines[0][0]

        result = []
        for i, line in enumerate(body_lines):
            if not line.strip():
                result.append("")
            elif i == first_idx:
                result.append(line.lstrip())
            else:
                current_indent = len(line) - len(line.lstrip())
                new_indent = max(0, current_indent - 4)
                result.append(" " * new_indent + line.lstrip())

        return result

    def _normalize_with_continuations(
        self, body_lines: list[str], non_empty_lines: list[tuple]
    ) -> list[str]:
        """Normalize blocks with elif/else/except/finally continuations."""
        continuation_keywords = ("elif ", "else:", "except ", "except:", "finally:")

        continuation_indents = []
        for _, line in non_empty_lines[1:]:
            stripped = line.strip()
            if any(stripped.startswith(kw) for kw in continuation_keywords):
                continuation_indents.append(len(line) - len(line.lstrip()))

        if not continuation_indents:
            return self._standard_normalize(body_lines, non_empty_lines)

        min_continuation = min(continuation_indents)
        first_idx = non_empty_lines[0][0]

        result = []
        for i, line in enumerate(body_lines):
            if not line.strip():
                result.append("")
            elif i == first_idx:
                result.append(line.lstrip())
            else:
                current_indent = len(line) - len(line.lstrip())
                new_indent = max(0, current_indent - min_continuation)
                result.append(" " * new_indent + line.lstrip())

        return result

    def _clean_answer(self, answer: str) -> str:
        """Clean answer text."""
        answer = answer.strip()

        if answer.startswith("```") and answer.endswith("```"):
            match = re.match(r"```(?:python)?\s*\n(.*?)```$", answer, re.DOTALL)
            if match:
                return match.group(1).strip()

        return answer

    def _extract_code_blocks(self, text: str) -> list[str]:
        """Extract code blocks from text."""
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches if match.strip()]

    def _replace_code_in_answer(self, answer: str, new_code: str) -> str:
        """Replace main code block in answer with corrected code."""
        code_blocks = self._extract_code_blocks(answer)
        if not code_blocks:
            return answer

        old_code = max(code_blocks, key=len)
        old_block = f"```python\n{old_code}\n```"
        new_block = f"```python\n{new_code}\n```"

        if old_block in answer:
            return answer.replace(old_block, new_block, 1)

        old_block_no_lang = f"```\n{old_code}\n```"
        if old_block_no_lang in answer:
            return answer.replace(old_block_no_lang, new_block, 1)

        return answer.replace(old_code, new_code, 1)


class AnswerBatchProcessor:
    """Processor for answer generation with full parallelization and checkpointing.

    All sessions run concurrently up to max_concurrent limit.
    Each session handles its own validation and correction loop independently.
    Supports incremental checkpointing for fault tolerance.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_concurrent: int = 20,
        tracer: Optional[GenerationTracer] = None,
    ):
        """Initialize batch processor.

        Args:
            llm_client: LLM client for generation
            max_concurrent: Maximum concurrent sessions
            tracer: Optional tracer for logging prompts and responses
        """
        self.llm_client = llm_client
        self.max_concurrent = max_concurrent
        self.tracer = tracer

    async def process_sessions_async(
        self,
        sessions: list[AnswerSession],
        progress_callback=None,
        checkpoint_callback=None,
        checkpoint_interval: int = 20,
        skip_indices: set[int] | None = None,
    ) -> list[AnswerResult]:
        """Process multiple answer sessions with full parallelization.

        All sessions run concurrently up to max_concurrent limit.
        Supports incremental checkpointing for fault tolerance.

        Args:
            sessions: List of answer sessions
            progress_callback: Optional callback(completed_count)
            checkpoint_callback: Optional callback(results_dict, completed_indices)
                                 for incremental saving
            checkpoint_interval: Save checkpoint every N completions
            skip_indices: Set of session indices to skip (already processed)

        Returns:
            List of AnswerResult (None for skipped indices)
        """
        if not sessions:
            return []

        skip_indices = skip_indices or set()
        results: list[AnswerResult | None] = [None] * len(sessions)
        completed_indices: set[int] = set()
        semaphore = asyncio.Semaphore(self.max_concurrent)
        completed_count = [0]
        last_checkpoint = [0]
        lock = asyncio.Lock()

        async def process_one(idx: int, session: AnswerSession) -> None:
            """Process single session with concurrency control."""
            if idx in skip_indices:
                return

            async with semaphore:
                result = await session.generate_and_validate_async()

            async with lock:
                results[idx] = result
                completed_indices.add(idx)
                completed_count[0] += 1

                if progress_callback:
                    progress_callback(completed_count[0])

                # Save checkpoint periodically
                if checkpoint_callback and (
                    completed_count[0] - last_checkpoint[0] >= checkpoint_interval
                ):
                    checkpoint_callback(results, completed_indices)
                    last_checkpoint[0] = completed_count[0]

        # Launch all concurrently
        tasks = [process_one(i, s) for i, s in enumerate(sessions)]
        await asyncio.gather(*tasks)

        # Final checkpoint
        if checkpoint_callback:
            checkpoint_callback(results, completed_indices)

        return results
