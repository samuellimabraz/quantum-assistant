"""Input planning for generating candidate inputs from chunks with improved multimodal strategy."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from synthetic_data.config import QuestionType
from synthetic_data.extractors.chunker import Chunk
from synthetic_data.models import LLMClient, Message
from synthetic_data.parsers.base import ImageReference, ImageType
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


@dataclass
class ChunkPlan:
    """Plan for generating inputs from a chunk."""

    chunk: Chunk
    text_candidates: int = 0
    image_candidates: list[tuple[int, QuestionType]] = field(default_factory=list)
    suitable_types: list[QuestionType] = field(default_factory=list)


@dataclass
class GenerationTask:
    """A task for batch generation."""

    chunk: Chunk
    question_type: QuestionType
    target_image: Optional[ImageReference] = None
    context: str = ""


class InputPlanner:
    """Plans and generates candidate inputs from chunks with batch optimization.

    For each chunk, the planner:
    1. Analyzes content to determine suitable question types
    2. Identifies images that could anchor multimodal questions
    3. Generates k candidate inputs using batch API
    4. Generates and validates tests for code types
    5. Filters candidates for quality
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompts: "PromptSet",
        candidates_per_chunk: int = 3,
        max_concurrent: int = 10,
        test_max_iterations: int = 3,
        tracer: Optional[GenerationTracer] = None,
    ):
        """Initialize input planner.

        Args:
            llm_client: LLM client for generation
            prompts: Prompt set for generation
            candidates_per_chunk: Number of candidates to generate per chunk
            max_concurrent: Maximum concurrent requests
            test_max_iterations: Max iterations for test correction loop
            tracer: Optional tracer for logging prompts and responses
        """
        self.llm_client = llm_client
        self.prompts = prompts
        self.candidates_per_chunk = candidates_per_chunk
        self.max_concurrent = max_concurrent
        self.test_max_iterations = test_max_iterations
        self.tracer = tracer

    def analyze_chunk(self, chunk: Chunk, multimodal_ratio: float = 0.5) -> ChunkPlan:
        """Analyze a chunk to determine generation potential.

        Returns a plan specifying how many candidates of each type to generate.

        Args:
            chunk: Content chunk to analyze
            multimodal_ratio: Target ratio of multimodal candidates (0.0-1.0)
        """
        plan = ChunkPlan(chunk=chunk)

        # Determine suitable question types based on content
        has_code = bool(chunk.code_blocks) or "```" in chunk.text
        has_concepts = any(
            kw in chunk.text.lower()
            for kw in [
                "quantum",
                "qubit",
                "circuit",
                "gate",
                "state",
                "superposition",
                "entanglement",
                "measurement",
            ]
        )

        if has_code:
            plan.suitable_types.extend(
                [QuestionType.FUNCTION_COMPLETION, QuestionType.CODE_GENERATION]
            )
        if has_concepts:
            plan.suitable_types.append(QuestionType.QA)

        if not plan.suitable_types:
            plan.suitable_types = [QuestionType.QA]

        # Filter images suitable for multimodal
        suitable_images = [
            img for img in chunk.transcribed_images if self._should_use_image_for_multimodal(img)
        ]

        # Plan multimodal and text-only candidates based on target ratio
        total_candidates = self.candidates_per_chunk

        if suitable_images and multimodal_ratio > 0:
            # Calculate multimodal candidates (at least 1 per suitable image, up to ratio)
            target_multimodal = max(1, int(total_candidates * multimodal_ratio))
            num_multimodal = min(target_multimodal, len(suitable_images))

            # Plan image-specific candidates
            for i in range(num_multimodal):
                if i < len(suitable_images):
                    img = suitable_images[i]
                    img_type = self._image_question_type(img)
                    plan.image_candidates.append((i, img_type))

            # Remaining candidates are text-only
            plan.text_candidates = total_candidates - num_multimodal
        else:
            # All text-only
            plan.text_candidates = total_candidates

        return plan

    def _image_question_type(self, img: ImageReference) -> QuestionType:
        """Determine best question type for an image based on its classified type."""
        # Use classified image type if available
        if img.image_type == ImageType.CIRCUIT:
            # Circuits are excellent for code generation (implement what you see)
            return QuestionType.CODE_GENERATION

        elif img.image_type == ImageType.CHART:
            # Charts/histograms are good for QA (analyze results)
            return QuestionType.QA

        elif img.image_type == ImageType.BLOCH_SPHERE:
            # Bloch spheres can be code or QA
            return QuestionType.CODE_GENERATION

        elif img.image_type == ImageType.DIAGRAM:
            # Diagrams are usually QA (explain the diagram)
            return QuestionType.QA

        elif img.image_type == ImageType.TABLE:
            # Tables are usually QA (interpret data)
            return QuestionType.QA

        elif img.image_type == ImageType.CODE_OUTPUT:
            # Code outputs are QA (analyze output)
            return QuestionType.QA

        elif img.image_type == ImageType.FORMULA:
            # Skip formulas for multimodal (usually decorative)
            return QuestionType.QA

        elif img.image_type == ImageType.DECORATIVE:
            # Skip decorative images
            return QuestionType.QA

        # Fallback: use transcription heuristics
        transcription = img.transcription.lower() if img.transcription else ""

        # Circuit keywords
        if any(kw in transcription for kw in ["circuit", "gate", "qubit", "hadamard", "cnot"]):
            return QuestionType.CODE_GENERATION

        # Graphs/charts -> QA (analysis)
        if any(kw in transcription for kw in ["histogram", "plot", "graph", "chart", "axis"]):
            return QuestionType.QA

        # Default to QA for explanatory content
        return QuestionType.QA

    def _should_use_image_for_multimodal(self, img: ImageReference) -> bool:
        """Determine if an image is suitable for multimodal generation.

        Filters out decorative and formula images that don't add value.
        """
        # Skip decorative images
        if img.image_type == ImageType.DECORATIVE:
            return False

        # Skip small formula images (usually not essential)
        if img.image_type == ImageType.FORMULA:
            # Only use if transcription is substantial
            if img.transcription and len(img.transcription) < 100:
                return False

        # Must have transcription
        if not img.transcription:
            return False

        # Good types for multimodal
        good_types = {
            ImageType.CIRCUIT,
            ImageType.CHART,
            ImageType.BLOCH_SPHERE,
            ImageType.DIAGRAM,
            ImageType.TABLE,
            ImageType.CODE_OUTPUT,
        }

        return img.image_type in good_types or img.image_type == ImageType.UNKNOWN

    async def generate_candidates_async(
        self,
        chunks: list[Chunk],
        question_type_weights: dict[QuestionType, float],
        multimodal_ratio: float = 0.5,
        progress_callback=None,
    ) -> list[InputCandidate]:
        """Generate candidate inputs from chunks using optimized batch processing.

        Args:
            chunks: Content chunks to process
            question_type_weights: Weights for question type distribution
            multimodal_ratio: Target ratio of multimodal candidates (0.0-1.0)
            progress_callback: Optional callback(completed_count)

        Returns:
            List of input candidates
        """
        # Phase 1: Prepare all generation tasks
        tasks = self._prepare_tasks(chunks, question_type_weights, multimodal_ratio)

        # Phase 2: Batch generate questions
        candidates = await self._batch_generate_questions_async(tasks, progress_callback)

        # Phase 3: Generate and validate tests for code types (with correction loop)
        candidates = await self._batch_generate_tests_async(candidates, progress_callback)

        return candidates

    def _prepare_tasks(
        self,
        chunks: list[Chunk],
        question_type_weights: dict[QuestionType, float],
        multimodal_ratio: float = 0.5,
    ) -> list[GenerationTask]:
        """Prepare all generation tasks from chunks."""
        tasks = []

        for chunk in chunks:
            plan = self.analyze_chunk(chunk, multimodal_ratio)

            # Text-only candidates
            for _ in range(plan.text_candidates):
                qtype = self._weighted_type_choice(plan.suitable_types, question_type_weights)
                context = chunk.build_context_with_transcriptions(include_code=True)
                tasks.append(
                    GenerationTask(
                        chunk=chunk,
                        question_type=qtype,
                        target_image=None,
                        context=context,
                    )
                )

            # Image-specific candidates (using suitable images only)
            suitable_images = [
                img
                for img in chunk.transcribed_images
                if self._should_use_image_for_multimodal(img)
            ]

            for img_idx, qtype in plan.image_candidates:
                if img_idx < len(suitable_images):
                    target_img = suitable_images[img_idx]
                    # Build context with target image emphasized
                    context = chunk.build_context_with_transcriptions(
                        target_image_id=target_img.image_id,
                        include_code=True,
                    )
                    tasks.append(
                        GenerationTask(
                            chunk=chunk,
                            question_type=qtype,
                            target_image=target_img,
                            context=context,
                        )
                    )

        return tasks

    async def _batch_generate_questions_async(
        self,
        tasks: list[GenerationTask],
        progress_callback=None,
    ) -> list[InputCandidate]:
        """Batch generate questions for all tasks."""
        if not tasks:
            return []

        # Build batch of message lists and trace conversations
        message_batches = []
        conversations = []

        for task in tasks:
            use_image = task.target_image is not None
            system_prompt = self.prompts.get_input_system_prompt(use_image=use_image)
            question_prompt = self.prompts.get_question_prompt(task.question_type).format(
                context=task.context
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
        for task, response, conv in zip(tasks, responses, conversations):
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
                        context=task.context,
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
                    context=task.context,
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
                if c.is_valid:  # Only process valid candidates
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
        semaphore = asyncio.Semaphore(self.max_concurrent)
        lock = asyncio.Lock()
        completed = [0]

        async def process_one(candidate: InputCandidate) -> InputCandidate:
            async with semaphore:
                result = await self._generate_test_with_correction_async(candidate)
                async with lock:
                    completed[0] += 1
                    if progress_callback:
                        progress_callback(completed[0])
                return result

        results = await asyncio.gather(*[process_one(c) for c in candidates])
        return list(results)

    async def _generate_test_with_correction_async(
        self,
        candidate: InputCandidate,
    ) -> InputCandidate:
        """Generate and validate test for a candidate with correction loop.

        Tests are mandatory for code types. If validation fails after max iterations,
        the candidate is rejected (marked invalid).
        """
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
        """Filter candidates for quality using batch API.

        Args:
            candidates: Candidates to filter
            filter_prompt: Prompt template for filtering
            system_prompt: System prompt for filtering
            progress_callback: Optional callback(completed_count)

        Returns:
            Filtered list of valid candidates
        """
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

    def _weighted_type_choice(
        self,
        suitable_types: list[QuestionType],
        weights: dict[QuestionType, float],
    ) -> QuestionType:
        """Choose a question type based on weights."""
        import random

        if not suitable_types:
            return QuestionType.QA

        type_weights = [(t, weights.get(t, 1.0)) for t in suitable_types]
        total = sum(w for _, w in type_weights)

        if total == 0:
            return random.choice(suitable_types)

        r = random.random() * total
        cumulative = 0
        for t, w in type_weights:
            cumulative += w
            if r <= cumulative:
                return t

        return suitable_types[-1]

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

        return None
