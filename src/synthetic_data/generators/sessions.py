"""Answer generation sessions with validation loop."""

import asyncio
import ast
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from synthetic_data.config import QuestionType
from synthetic_data.generators.planner import InputCandidate
from synthetic_data.models import LLMClient, Message
from synthetic_data.utils.tracer import GenerationTracer, ConversationTrace


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
    ):
        """Initialize answer session.

        Args:
            llm_client: LLM client for generation
            system_prompt: System prompt for the session
            answer_prompt: User prompt for initial answer
            correction_prompt: Prompt template for corrections
            candidate: Input candidate being answered
            max_iterations: Maximum correction attempts
            timeout_seconds: Timeout for test execution
            tracer: Optional tracer for logging prompts and responses
        """
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.answer_prompt = answer_prompt
        self.correction_prompt = correction_prompt
        self.candidate = candidate
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.tracer = tracer
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
                corrected = await self.llm_client.generate_async(self.messages, temperature=0.1)

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

        # Log the test code being used
        if self.tracer and self._conversation:
            self.tracer.log_entry(
                stage="answer_generation",
                step_type="test_code",
                content=self.candidate.test_code,
                candidate_id=self._conversation.id,
                iteration=0,
            )

        for iteration in range(self.max_iterations):
            test_result = await self._run_test_async(current_answer)

            # Log validation result
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
                corrected = await self.llm_client.generate_async(self.messages, temperature=0.1)

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

            error_lines = stderr_text.split("\n")
            relevant_error = next(
                (
                    line
                    for line in reversed(error_lines)
                    if "Error:" in line or "assert" in line.lower()
                ),
                stderr_text[-500:] if stderr_text else "Unknown error",
            )
            return ValidationResult(
                passed=False,
                error_type="test_failure",
                error_message=relevant_error,
            )

        except (OSError, IOError) as e:
            return ValidationResult(
                passed=False,
                error_type="execution",
                error_message=str(e),
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)

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

        Handles two common model output patterns:

        1. Problematic pattern (same-level lines with wrong base):
           - First line has 0 indentation
           - Subsequent lines have 4+ indentation
           - First line is NOT a block starter
           => All lines should be at same level (0), strip the base

        2. Block starter pattern:
           - First line ends with `:` (if/for/while/def/class/etc)
           - Subsequent lines are indented
           => Keep first line at 0, normalize subsequent to have 4-space indent

        Returns lines with normalized relative indentation.
        """
        non_empty_lines = [(i, line) for i, line in enumerate(body_lines) if line.strip()]

        if not non_empty_lines:
            return body_lines

        first_idx, first_line = non_empty_lines[0]
        first_indent = len(first_line) - len(first_line.lstrip())
        first_stripped = first_line.strip()

        # Check if first line is a block starter
        is_block_start = first_stripped.endswith(":")

        if first_indent == 0 and len(non_empty_lines) > 1:
            subsequent_indents = [len(line) - len(line.lstrip()) for _, line in non_empty_lines[1:]]

            if subsequent_indents:
                min_subsequent = min(subsequent_indents)

                if is_block_start and min_subsequent > 0:
                    # Block starter: normalize subsequent lines to have proper indent
                    # If min_subsequent is 8, we want 4 (one level of indentation)
                    # Calculate excess indent (subtract one level = 4 spaces)
                    excess = max(0, min_subsequent - 4)
                    result = []
                    for i, line in enumerate(body_lines):
                        if not line.strip():
                            result.append("")
                        elif i == first_idx:
                            result.append(line.lstrip())
                        else:
                            current_indent = len(line) - len(line.lstrip())
                            normalized = max(0, current_indent - excess)
                            result.append(" " * normalized + line.lstrip())
                    return result

                elif not is_block_start and min_subsequent > 0:
                    # Same-level pattern: all lines should be at same base level
                    result = []
                    for i, line in enumerate(body_lines):
                        if not line.strip():
                            result.append("")
                        elif i == first_idx:
                            result.append(line.lstrip())
                        else:
                            current_indent = len(line) - len(line.lstrip())
                            relative = max(0, current_indent - min_subsequent)
                            result.append(" " * relative + line.lstrip())
                    return result

        # Standard normalization: subtract minimum indent from all lines
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
    """Batch processor for answer generation sessions."""

    def __init__(
        self,
        llm_client: LLMClient,
        max_concurrent: int = 10,
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
    ) -> list[AnswerResult]:
        """Process multiple answer sessions concurrently.

        Args:
            sessions: List of answer sessions
            progress_callback: Optional callback(completed_count)

        Returns:
            List of AnswerResult
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        completed = [0]
        lock = asyncio.Lock()

        async def process_one(session: AnswerSession) -> AnswerResult:
            async with semaphore:
                result = await session.generate_and_validate_async()
                async with lock:
                    completed[0] += 1
                    if progress_callback:
                        progress_callback(completed[0])
                return result

        results = await asyncio.gather(*[process_one(s) for s in sessions])
        return list(results)
