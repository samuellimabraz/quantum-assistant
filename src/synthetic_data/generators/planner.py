"""Input planning for generating candidate inputs from chunks."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from synthetic_data.config import QuestionType
from synthetic_data.extractors.chunker import Chunk
from synthetic_data.generators.allocation import (
    AllocationConfig,
    AllocationResult,
    Allocator,
    SampleTask,
)
from synthetic_data.generators.allocation import AllocationMetrics
from synthetic_data.models import LLMClient, Message
from synthetic_data.parsers.base import ImageReference
from synthetic_data.utils.tracer import GenerationTracer

if TYPE_CHECKING:
    from synthetic_data.generators.prompts import PromptSet


@dataclass
class InputCandidate:
    """A candidate input generated from a chunk."""

    chunk: Chunk
    question: str
    question_type: QuestionType
    test_code: Optional[str] = None
    entry_point: Optional[str] = None
    target_image: Optional[ImageReference] = None
    context: str = ""
    score: float = 0.0
    rejection_reason: Optional[str] = None

    @property
    def is_multimodal(self) -> bool:
        """Check if this candidate uses an image."""
        return self.target_image is not None

    @property
    def is_valid(self) -> bool:
        """Check if candidate passed filtering."""
        return self.rejection_reason is None and self.question


class InputPlanner:
    """Plans and generates candidate inputs from chunks.

    Uses content-aware allocation strategy that:
    1. Creates unique (chunk, image, type) combinations
    2. Respects per-type multimodal ratios
    3. Selects candidates by quality score
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompts: "PromptSet",
        max_concurrent: int = 10,
        test_max_iterations: int = 3,
        tracer: Optional[GenerationTracer] = None,
    ):
        """Initialize input planner.

        Args:
            llm_client: LLM client for generation
            prompts: Prompt set for generation
            max_concurrent: Maximum concurrent requests
            test_max_iterations: Max iterations for test correction loop
            tracer: Optional tracer for logging prompts and responses
        """
        self.llm_client = llm_client
        self.prompts = prompts
        self.max_concurrent = max_concurrent
        self.test_max_iterations = test_max_iterations
        self.tracer = tracer

    async def generate_candidates_async(
        self,
        chunks: list[Chunk],
        allocation_config: AllocationConfig,
        progress_callback=None,
        diversity_weight: float = 0.4,
    ) -> tuple[list[InputCandidate], AllocationResult]:
        """Generate candidate inputs from chunks using allocation config.

        Args:
            chunks: Content chunks to process
            allocation_config: Configuration for allocation
            progress_callback: Optional callback(completed_count)
            diversity_weight: Weight for diversity vs score in selection

        Returns:
            Tuple of (candidates list, allocation result)
        """
        # Phase 1: Allocate chunks to tasks with diversity awareness
        allocator = Allocator(allocation_config, diversity_weight=diversity_weight)
        allocation_result = allocator.allocate(chunks)

        # Log allocation statistics
        print(f"    Allocated {len(allocation_result.tasks)} tasks")
        print(f"    Multimodal: {allocation_result.multimodal_samples}")
        print(f"    Text-only: {allocation_result.text_only_samples}")
        print("    By question type:")
        by_type = allocation_result.samples_by_type()
        mm_by_type = allocation_result.multimodal_by_type()
        for qt in QuestionType:
            total = by_type[qt]
            mm = mm_by_type[qt]
            if total > 0:
                print(f"      {qt.value}: {total} (multimodal: {mm})")

        # Phase 2: Build context for each task
        tasks = self._build_generation_tasks(allocation_result.tasks)

        # Phase 3: Batch generate questions
        candidates = await self._batch_generate_questions_async(tasks, progress_callback)

        # Phase 4: Generate and validate tests for code types
        candidates = await self._batch_generate_tests_async(candidates, progress_callback)

        return candidates, allocation_result

    def _build_generation_tasks(
        self, sample_tasks: list[SampleTask]
    ) -> list[tuple[SampleTask, str]]:
        """Build context strings for each sample task.

        Returns list of (task, context) tuples.
        """
        tasks_with_context = []

        for task in sample_tasks:
            context = task.chunk.build_context_with_transcriptions(
                target_image_id=task.target_image.image_id if task.target_image else None,
                include_code=True,
            )
            tasks_with_context.append((task, context))

        return tasks_with_context

    async def _batch_generate_questions_async(
        self,
        tasks: list[tuple[SampleTask, str]],
        progress_callback=None,
    ) -> list[InputCandidate]:
        """Batch generate questions for all tasks."""
        if not tasks:
            return []

        # Build batch of message lists
        message_batches = []
        conversations = []

        for task, context in tasks:
            use_image = task.is_multimodal
            system_prompt = self.prompts.get_input_system_prompt(use_image=use_image)
            question_prompt = self.prompts.get_question_prompt(task.question_type).format(
                context=context
            )
            message_batches.append(
                [
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=question_prompt),
                ]
            )

            # Start conversation trace
            if self.tracer:
                conv = self.tracer.start_conversation(
                    stage="input_generation",
                    question_type=task.question_type.value,
                    is_multimodal=use_image,
                    chunk_id=task.chunk.chunk_id,
                    score=task.score,
                )
                self.tracer.log_message(conv, "system", system_prompt)
                self.tracer.log_message(conv, "user", question_prompt)
                conversations.append(conv)
            else:
                conversations.append(None)

        # Batch generate all questions
        responses = await self.llm_client.generate_batch_async(
            message_batches,
            max_concurrent=self.max_concurrent,
            temperature=0.7,
            progress_callback=progress_callback,
        )

        # Build candidates from responses
        candidates = []
        for (task, context), response, conv in zip(tasks, responses, conversations):
            question = response.strip() if response else ""

            # Log response
            if self.tracer and conv:
                self.tracer.log_message(conv, "assistant", response or "(empty)")
                self.tracer.complete_conversation(
                    conv,
                    success=bool(question),
                    total_iterations=1,
                )

            if not question:
                candidates.append(
                    InputCandidate(
                        chunk=task.chunk,
                        question="",
                        question_type=task.question_type,
                        target_image=task.target_image,
                        context=context,
                        score=task.score,
                        rejection_reason="Empty question generated",
                    )
                )
                continue

            # Extract entry point for code types
            entry_point = None
            if task.question_type in (
                QuestionType.FUNCTION_COMPLETION,
                QuestionType.CODE_GENERATION,
            ):
                entry_point = self._extract_entry_point(question)
                if not entry_point:
                    entry_point = "solution"

            candidates.append(
                InputCandidate(
                    chunk=task.chunk,
                    question=question,
                    question_type=task.question_type,
                    entry_point=entry_point,
                    target_image=task.target_image,
                    context=context,
                    score=task.score,
                )
            )

        return candidates

    async def _batch_generate_tests_async(
        self,
        candidates: list[InputCandidate],
        progress_callback=None,
    ) -> list[InputCandidate]:
        """Generate and validate tests for code type candidates.

        For code types (function_completion, code_generation), tests are MANDATORY.
        If test generation or validation fails after max iterations, candidate is rejected.
        """
        # Separate code and non-code candidates
        code_candidates = []
        non_code_candidates = []

        for c in candidates:
            if c.question_type in (QuestionType.FUNCTION_COMPLETION, QuestionType.CODE_GENERATION):
                if c.is_valid:
                    code_candidates.append(c)
                else:
                    non_code_candidates.append(c)
            else:
                non_code_candidates.append(c)

        if not code_candidates:
            return non_code_candidates

        # Process code candidates with test generation and validation
        processed = await self._generate_and_validate_tests_async(
            code_candidates, progress_callback
        )

        return non_code_candidates + processed

    async def _generate_and_validate_tests_async(
        self,
        candidates: list[InputCandidate],
        progress_callback=None,
    ) -> list[InputCandidate]:
        """Generate tests with validation and correction loop."""
        if not candidates:
            return []

        semaphore = asyncio.Semaphore(self.max_concurrent)
        completed = [0]
        lock = asyncio.Lock()

        async def process_one(candidate: InputCandidate) -> InputCandidate:
            """Process single candidate with concurrency control."""
            async with semaphore:
                result = await self._generate_test_with_correction_async(candidate)

            async with lock:
                completed[0] += 1
                if progress_callback:
                    progress_callback(completed[0])

            return result

        tasks = [process_one(c) for c in candidates]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def _generate_test_with_correction_async(
        self,
        candidate: InputCandidate,
    ) -> InputCandidate:
        """Generate and validate test for a candidate with correction loop."""
        # Start conversation trace
        conv = None
        if self.tracer:
            conv = self.tracer.start_conversation(
                stage="test_generation",
                question_type=candidate.question_type.value,
                entry_point=candidate.entry_point,
                is_multimodal=candidate.is_multimodal,
            )

        # Build conversation for test generation
        use_image = candidate.is_multimodal
        system_prompt = self.prompts.get_input_system_prompt(use_image=use_image)
        question_prompt = self.prompts.get_question_prompt(candidate.question_type).format(
            context=candidate.context
        )

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=question_prompt),
            Message(role="assistant", content=candidate.question),
        ]

        # Log initial context
        if self.tracer and conv:
            self.tracer.log_message(conv, "system", system_prompt, iteration=0)
            self.tracer.log_message(
                conv, "user", question_prompt, iteration=0, step="question_context"
            )
            self.tracer.log_message(
                conv, "assistant", candidate.question, iteration=0, step="generated_question"
            )

        # Generate initial test
        test_prompt = self.prompts.test_generation_prompt.format(
            question=candidate.question,
            entry_point=candidate.entry_point,
        )
        messages.append(Message(role="user", content=test_prompt))

        if self.tracer and conv:
            self.tracer.log_message(conv, "user", test_prompt, iteration=1, step="test_request")

        try:
            test_response = await self.llm_client.generate_async(messages, temperature=0.2)
            test_code = self._extract_code(test_response)

            if self.tracer and conv:
                self.tracer.log_message(
                    conv, "assistant", test_response, iteration=1, step="test_response"
                )
        except Exception as e:
            if self.tracer and conv:
                self.tracer.complete_conversation(conv, success=False, error=str(e))
            candidate.rejection_reason = f"Test generation failed: {e}"
            return candidate

        if not test_code:
            if self.tracer and conv:
                self.tracer.complete_conversation(conv, success=False, error="Empty test code")
            candidate.rejection_reason = "Empty test code generated"
            return candidate

        messages.append(Message(role="assistant", content=test_response))

        # Validation and correction loop
        for iteration in range(self.test_max_iterations):
            validation_error = self._validate_test_structure(test_code, candidate.entry_point)

            if self.tracer and conv:
                self.tracer.log_entry(
                    stage="test_generation",
                    step_type="validation",
                    content=(
                        f"Validation error: {validation_error}"
                        if validation_error
                        else "Validation passed"
                    ),
                    candidate_id=conv.id,
                    iteration=iteration + 1,
                )

            if validation_error is None:
                # Test is valid
                candidate.test_code = test_code
                if self.tracer and conv:
                    self.tracer.complete_conversation(
                        conv,
                        success=True,
                        total_iterations=iteration + 1,
                        final_test_code=test_code,
                    )
                return candidate

            # Test failed validation, request correction
            if iteration < self.test_max_iterations - 1:
                correction_prompt = (
                    f"The test has an error: {validation_error}\n\n"
                    f"Please fix the test. The test MUST:\n"
                    f"1. Have valid Python syntax\n"
                    f"2. Contain a check(candidate) function OR test_* function\n"
                    f"3. Include assert statements\n"
                    f"4. Call the function '{candidate.entry_point}'\n\n"
                    f"Return ONLY the corrected Python code block."
                )
                messages.append(Message(role="user", content=correction_prompt))

                if self.tracer and conv:
                    self.tracer.log_message(
                        conv,
                        "user",
                        correction_prompt,
                        iteration=iteration + 2,
                        step="correction_request",
                        validation_error=validation_error,
                    )

                try:
                    corrected_response = await self.llm_client.generate_async(
                        messages, temperature=0.1
                    )
                    corrected_code = self._extract_code(corrected_response)

                    if self.tracer and conv:
                        self.tracer.log_message(
                            conv,
                            "assistant",
                            corrected_response,
                            iteration=iteration + 2,
                            step="correction_response",
                        )

                    if corrected_code:
                        test_code = corrected_code
                        messages.append(Message(role="assistant", content=corrected_response))
                    else:
                        break
                except Exception as e:
                    if self.tracer and conv:
                        self.tracer.log_entry(
                            stage="test_generation",
                            step_type="error",
                            content=f"Correction failed: {e}",
                            candidate_id=conv.id,
                            iteration=iteration + 2,
                        )
                    break

        # All iterations exhausted, test is still invalid
        if self.tracer and conv:
            self.tracer.complete_conversation(
                conv,
                success=False,
                total_iterations=self.test_max_iterations,
                final_error=f"Test validation failed after {self.test_max_iterations} attempts",
            )
        candidate.rejection_reason = (
            f"Test validation failed after {self.test_max_iterations} attempts"
        )
        return candidate

    async def filter_candidates_async(
        self,
        candidates: list[InputCandidate],
        filter_prompt: str,
        system_prompt: str,
        progress_callback=None,
    ) -> list[InputCandidate]:
        """Filter candidates for quality using batch API."""
        valid_candidates = [c for c in candidates if c.is_valid]

        if not valid_candidates:
            return []

        # Prepare filter requests as batch
        filter_batches = []
        for candidate in valid_candidates:
            prompt = filter_prompt.format(
                question=candidate.question,
                question_type=candidate.question_type.value,
                has_image="yes" if candidate.is_multimodal else "no",
                has_test="yes" if candidate.test_code else "no",
                context_preview=(
                    candidate.context[:500] + "..."
                    if len(candidate.context) > 500
                    else candidate.context
                ),
            )
            filter_batches.append(
                [
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=prompt),
                ]
            )

        # Run filter as batch
        responses = await self.llm_client.generate_batch_async(
            filter_batches,
            max_concurrent=self.max_concurrent,
            temperature=0.1,
            progress_callback=progress_callback,
        )

        # Parse responses
        filtered = []
        for candidate, response in zip(valid_candidates, responses):
            decision, reason = self._parse_filter_response(response)
            if decision == "PASS":
                filtered.append(candidate)
            else:
                candidate.rejection_reason = reason

        return filtered

    def _parse_filter_response(self, response: str) -> tuple[str, str]:
        """Parse filter response to get decision and reason."""
        response = response.strip().upper()

        if "PASS" in response or "YES" in response:
            return "PASS", ""

        # Extract reason if REJECT
        reason = "Quality check failed"
        lines = response.split("\n")
        for line in lines:
            if "REASON" in line.upper() and ":" in line:
                reason = line.split(":", 1)[1].strip()
                break

        return "REJECT", reason

    def _extract_entry_point(self, question: str) -> Optional[str]:
        """Extract function name from question."""
        patterns = [
            r"function\s+named\s+[`'\"]([a-zA-Z_][a-zA-Z0-9_]*)[`'\"]",
            r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
            r"named\s+`([a-zA-Z_][a-zA-Z0-9_]*)`",
            r"`([a-zA-Z_][a-zA-Z0-9_]*)`\s+with",
        ]

        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                name = match.group(1)
                if name.lower() not in ("name", "that", "this", "function", "it"):
                    return name

        return None

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract code from response."""
        code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
        if code_blocks:
            return max(code_blocks, key=len).strip()
        return None

    def _validate_test_structure(self, test_code: str, entry_point: str) -> Optional[str]:
        """Validate test code structure. Returns error message or None if valid."""
        import ast

        try:
            ast.parse(test_code)
        except SyntaxError as e:
            return f"Syntax error: {e.msg}"

        has_check = re.search(r"def\s+check\s*\(", test_code)
        has_test = re.search(r"def\s+test_", test_code)
        if not has_check and not has_test:
            return "Test must contain check(candidate) or test_ function"

        if not re.search(r"assert\s+", test_code):
            return "Test must contain assertions"

        if entry_point not in test_code:
            return f"Test must call '{entry_point}'"

        # Check for deprecated/unavailable Qiskit APIs
        deprecated_error = self._check_deprecated_apis(test_code)
        if deprecated_error:
            return deprecated_error

        return None

    def _check_deprecated_apis(self, test_code: str) -> Optional[str]:
        """Check for deprecated or unavailable Qiskit APIs in test code.

        Returns error message if deprecated APIs found, None otherwise.
        """
        # Known deprecated/removed APIs in Qiskit 2.0
        deprecated_patterns = [
            (r"from\s+qiskit\.test", "qiskit.test module removed in Qiskit 2.0"),
            (
                r"from\s+qiskit\.providers\.fake_provider\s+import\s+.*FakeVigo",
                "FakeVigo removed - use FakeGeneric or qiskit_ibm_runtime.fake_provider",
            ),
            (
                r"from\s+qiskit\.primitives\s+import\s+.*BaseEstimator",
                "BaseEstimator removed - use StatevectorEstimator directly",
            ),
            (
                r"from\s+qiskit\.circuit\.library\s+import\s+.*random_clifford",
                "random_clifford moved to qiskit.quantum_info.random_clifford",
            ),
            (r"\.bind_parameters\s*\(", "bind_parameters deprecated - use assign_parameters"),
            (
                r"qiskit\.execute\s*\(",
                "qiskit.execute removed - use primitives (Sampler/Estimator)",
            ),
            (
                r"from\s+qiskit\s+import\s+.*execute",
                "execute removed - use primitives (Sampler/Estimator)",
            ),
            (
                r"\.operation\.qubits",
                "Wrong API: use circuit_instruction.qubits not .operation.qubits",
            ),
            (
                r"\.operation\.clbits",
                "Wrong API: use circuit_instruction.clbits not .operation.clbits",
            ),
        ]

        for pattern, message in deprecated_patterns:
            if re.search(pattern, test_code):
                return f"Deprecated API: {message}"

        return None
