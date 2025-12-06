"""Plan stage - Generate questions and tests from chunks."""

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from synthetic_data.config import PipelineConfig, QuestionType
from synthetic_data.extractors.chunker import Chunk
from synthetic_data.generators.allocation import (
    AllocationConfig,
    AllocationMetrics,
    AllocationResult,
    Allocator,
    SampleTask,
    TypeAllocationConfig,
)
from synthetic_data.generators.prompts import PromptSet
from synthetic_data.generators.types import InputCandidate
from synthetic_data.models import LLMClient, Message, ModelRegistry
from synthetic_data.utils import CheckpointManager

console = Console()


@dataclass
class PlanCheckpoint:
    """Checkpoint state for planning stage."""

    tasks: list[SampleTask] = field(default_factory=list)
    candidates: list[InputCandidate] = field(default_factory=list)
    processed_indices: list[int] = field(default_factory=list)
    phase: str = "questions"  # "questions", "tests", "complete"


class PlanStage:
    """Plan stage - generates questions and tests from chunks.

    Input: filtered/chunks.pkl
    Output: planned/candidates.pkl
    """

    stage_name = "planned"

    def __init__(
        self,
        config: PipelineConfig,
        model_registry: ModelRegistry,
        base_dir: Path,
        no_cache: bool = False,
    ):
        self.config = config
        self.gen_config = config.generation
        self.model_registry = model_registry
        self.base_dir = Path(base_dir)
        self.no_cache = no_cache

        self.output_dir = self.base_dir / self.stage_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.base_dir / ".checkpoints"
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir, "plan")

        self.prompts = self._build_prompts()

    def _build_prompts(self) -> PromptSet:
        """Build prompt set from config."""
        return PromptSet(
            input_generation_system=self.config.prompts.input_generation_system,
            function_completion_prompt=self.config.prompts.function_completion_prompt,
            code_generation_prompt=self.config.prompts.code_generation_prompt,
            qa_prompt=self.config.prompts.qa_prompt,
            test_generation_prompt=self.config.prompts.test_generation_prompt,
        )

    @property
    def output_file(self) -> Path:
        return self.output_dir / "candidates.pkl"

    def get_input_path(self) -> Path:
        """Get input path - filtered chunks."""
        filtered_dir = self.base_dir / "filtered"
        if (filtered_dir / "chunks.pkl").exists():
            return filtered_dir / "chunks.pkl"

        chunks_dir = self.base_dir / "chunks"
        if (chunks_dir / "chunks.pkl").exists():
            return chunks_dir / "chunks.pkl"

        raise FileNotFoundError("No chunks found. Run 'chunk' or 'filter-chunks' first.")

    def run(self, progress_callback=None) -> list[InputCandidate]:
        """Run the planning stage."""
        return asyncio.run(self.run_async(progress_callback))

    async def run_async(self, progress_callback=None) -> list[InputCandidate]:
        """Run planning stage asynchronously."""
        # Check if output exists
        if self.output_file.exists() and not self.no_cache:
            console.print(f"[cyan]Loading existing candidates: {self.output_file}[/cyan]")
            import pickle

            with open(self.output_file, "rb") as f:
                return pickle.load(f)

        # Load input
        console.print(f"[cyan]Loading chunks from: {self.get_input_path()}[/cyan]")
        import pickle

        with open(self.get_input_path(), "rb") as f:
            chunks = pickle.load(f)

        console.print(f"[cyan]Loaded {len(chunks)} chunks[/cyan]")

        # Load checkpoint if exists
        checkpoint = self._load_checkpoint()

        # Build allocation config
        allocation_config = self._build_allocation_config()

        # Log targets
        console.print(f"[cyan]Target: {self.gen_config.target_samples} samples[/cyan]")
        console.print(
            f"[cyan]Over-allocation factor: {allocation_config.over_allocation_factor}x[/cyan]"
        )

        # Get LLM client
        llm_client = self.model_registry.get_llm_client(self.gen_config.question_model)

        try:
            candidates, metrics = await self._generate_candidates_async(
                chunks, allocation_config, llm_client, checkpoint, progress_callback
            )
        finally:
            await llm_client.aclose()

        # Filter to valid candidates only
        valid_candidates = [c for c in candidates if c.is_valid]

        # Save output
        with open(self.output_file, "wb") as f:
            pickle.dump(valid_candidates, f)

        # Clear checkpoint
        self.checkpoint_manager.clear_checkpoint()

        # Print summary
        self._print_summary(valid_candidates, len(candidates) - len(valid_candidates), metrics)

        return valid_candidates

    async def _generate_candidates_async(
        self,
        chunks: list[Chunk],
        allocation_config: AllocationConfig,
        llm_client: LLMClient,
        checkpoint: Optional[PlanCheckpoint],
        progress_callback=None,
    ) -> tuple[list[InputCandidate], AllocationMetrics]:
        """Generate candidates from chunks."""
        # Allocate tasks
        if checkpoint and checkpoint.tasks:
            tasks = checkpoint.tasks
            allocation_result = AllocationResult(
                tasks=tasks,
                metrics=AllocationMetrics(
                    total_chunks=len(set(t.chunk_key for t in tasks)),
                    chunks_used=len(set(t.chunk_key for t in tasks)),
                ),
            )
            console.print(f"[cyan]Resuming with {len(tasks)} tasks from checkpoint[/cyan]")
        else:
            allocator = Allocator(
                allocation_config, diversity_weight=self.gen_config.diversity_weight
            )
            allocation_result = allocator.allocate(chunks)
            tasks = allocation_result.tasks

            console.print(f"[cyan]Allocated {len(tasks)} tasks[/cyan]")
            self._log_allocation(allocation_result)

        # Build contexts
        tasks_with_context = []
        for task in tasks:
            context = task.chunk.build_context_with_transcriptions(
                target_image_id=task.target_image.image_id if task.target_image else None,
                include_code=True,
            )
            tasks_with_context.append((task, context))

        # Estimate total work (questions + expected tests for code types)
        code_task_count = sum(
            1
            for task, _ in tasks_with_context
            if task.question_type
            in (QuestionType.FUNCTION_COMPLETION, QuestionType.CODE_GENERATION)
        )
        total_work = len(tasks_with_context) + code_task_count

        # Set progress total
        if progress_callback and hasattr(progress_callback, "set_total"):
            progress_callback.set_total(total_work)

        # Phase 1: Generate questions
        skip_indices = set(checkpoint.processed_indices) if checkpoint else set()
        existing_candidates = list(checkpoint.candidates) if checkpoint else []

        completed_count = [0]

        def question_callback(completed: int):
            completed_count[0] = completed
            if progress_callback:
                if hasattr(progress_callback, "update"):
                    progress_callback.update(completed)
                else:
                    progress_callback(completed)

        candidates = await self._generate_questions_async(
            tasks_with_context,
            llm_client,
            skip_indices,
            existing_candidates,
            tasks,
            question_callback,
        )

        # Phase 2: Generate tests for code candidates
        code_candidates = [
            (i, c)
            for i, c in enumerate(candidates)
            if c.is_valid
            and c.question_type in (QuestionType.FUNCTION_COMPLETION, QuestionType.CODE_GENERATION)
        ]

        if code_candidates:
            base_progress = completed_count[0]

            def test_callback(completed: int):
                total_completed = base_progress + completed
                if progress_callback:
                    if hasattr(progress_callback, "update"):
                        progress_callback.update(total_completed)
                    else:
                        progress_callback(total_completed)

            candidates = await self._generate_tests_async(
                candidates,
                code_candidates,
                llm_client,
                tasks,
                test_callback,
            )

        return candidates, allocation_result.metrics

    async def _generate_questions_async(
        self,
        tasks_with_context: list[tuple[SampleTask, str]],
        llm_client: LLMClient,
        skip_indices: set[int],
        existing_candidates: list[InputCandidate],
        tasks: list[SampleTask],
        progress_callback=None,
    ) -> list[InputCandidate]:
        """Generate questions for all tasks."""
        candidates: list[Optional[InputCandidate]] = [None] * len(tasks_with_context)

        # Fill existing
        for idx, candidate in enumerate(existing_candidates):
            if idx < len(candidates):
                candidates[idx] = candidate

        # Build message batches
        task_indices = []
        message_batches = []

        for idx, (task, context) in enumerate(tasks_with_context):
            if idx in skip_indices:
                continue

            task_indices.append(idx)
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

        if not message_batches:
            return [c for c in candidates if c is not None]

        # Set total for progress tracking
        if progress_callback and hasattr(progress_callback, "set_total"):
            progress_callback.set_total(len(message_batches))

        # Track progress with checkpoint saving
        checkpoint_interval = 20
        last_checkpoint = [0]

        def update_callback(completed: int):
            if progress_callback:
                if hasattr(progress_callback, "update"):
                    progress_callback.update(completed)
                else:
                    progress_callback(completed)

            # Save checkpoint periodically
            if completed - last_checkpoint[0] >= checkpoint_interval or completed == len(
                message_batches
            ):
                processed_so_far = set(skip_indices)
                for i in range(completed):
                    idx = task_indices[i]
                    processed_so_far.add(idx)

                checkpoint_state = PlanCheckpoint(
                    tasks=tasks,
                    candidates=[c for c in candidates if c is not None],
                    processed_indices=list(processed_so_far),
                    phase="questions",
                )
                self._save_checkpoint(checkpoint_state)
                last_checkpoint[0] = completed

        responses = await llm_client.generate_batch_async(
            message_batches,
            max_concurrent=self.gen_config.llm_concurrency,
            temperature=1.0,
            progress_callback=update_callback,
        )

        # Process responses
        for resp_idx, response in enumerate(responses):
            task_idx = task_indices[resp_idx]
            task, context = tasks_with_context[task_idx]

            question = response.strip() if response else ""

            if not question:
                candidate = InputCandidate(
                    chunk=task.chunk,
                    question="",
                    question_type=task.question_type,
                    target_image=task.target_image,
                    context=context,
                    score=task.score,
                    rejection_reason="Empty question generated",
                )
            else:
                entry_point = None
                if task.question_type in (
                    QuestionType.FUNCTION_COMPLETION,
                    QuestionType.CODE_GENERATION,
                ):
                    entry_point = self._extract_entry_point(question) or "solution"

                candidate = InputCandidate(
                    chunk=task.chunk,
                    question=question,
                    question_type=task.question_type,
                    entry_point=entry_point,
                    target_image=task.target_image,
                    context=context,
                    score=task.score,
                )

            candidates[task_idx] = candidate

        # Save final checkpoint (covered by update_callback at completion)

        return [c for c in candidates if c is not None]

    async def _generate_tests_async(
        self,
        candidates: list[InputCandidate],
        code_candidates: list[tuple[int, InputCandidate]],
        llm_client: LLMClient,
        tasks: list[SampleTask],
        progress_callback=None,
    ) -> list[InputCandidate]:
        """Generate tests for code-type candidates."""
        # Build test generation messages
        test_batches = []
        test_indices = []

        for idx, candidate in code_candidates:
            if candidate.test_code:  # Already has test
                continue

            test_indices.append(idx)
            use_image = candidate.is_multimodal
            system_prompt = self.prompts.get_input_system_prompt(use_image=use_image)
            question_prompt = self.prompts.get_question_prompt(candidate.question_type).format(
                context=candidate.context
            )
            test_prompt = self.prompts.test_generation_prompt.format(
                question=candidate.question,
                entry_point=candidate.entry_point,
            )

            test_batches.append(
                [
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=question_prompt),
                    Message(role="assistant", content=candidate.question),
                    Message(role="user", content=test_prompt),
                ]
            )

        if not test_batches:
            return candidates

        # Track for checkpoint saving
        checkpoint_interval = 20
        last_checkpoint = [0]

        def test_callback(completed: int):
            if progress_callback:
                progress_callback(completed)

            # Save checkpoint periodically
            if completed - last_checkpoint[0] >= checkpoint_interval or completed == len(
                test_batches
            ):
                checkpoint_state = PlanCheckpoint(
                    tasks=tasks,
                    candidates=[c for c in candidates if c is not None],
                    processed_indices=list(range(len(candidates))),
                    phase="tests",
                )
                self._save_checkpoint(checkpoint_state)
                last_checkpoint[0] = completed

        responses = await llm_client.generate_batch_async(
            test_batches,
            max_concurrent=self.gen_config.llm_concurrency,
            temperature=1.0,
            progress_callback=test_callback,
        )

        # Process test responses
        for resp_idx, response in enumerate(responses):
            candidate_idx = test_indices[resp_idx]
            candidate = candidates[candidate_idx]

            test_code = self._extract_code(response)
            if test_code:
                validation_error = self._validate_test_structure(test_code, candidate.entry_point)
                if validation_error is None:
                    candidates[candidate_idx].test_code = test_code
                else:
                    candidates[candidate_idx].rejection_reason = f"Invalid test: {validation_error}"
            else:
                candidates[candidate_idx].rejection_reason = "Empty test code generated"

        return candidates

    def _build_allocation_config(self) -> AllocationConfig:
        """Build allocation config from generation config."""
        type_configs = {}

        if self.gen_config.type_allocations:
            for type_name, type_cfg in self.gen_config.type_allocations.items():
                try:
                    qt = QuestionType(type_name)
                    type_configs[qt] = TypeAllocationConfig(
                        ratio=type_cfg.ratio,
                        multimodal_ratio=type_cfg.multimodal_ratio,
                    )
                except ValueError:
                    pass

        if not type_configs:
            type_configs = {
                QuestionType.QA: TypeAllocationConfig(ratio=0.30, multimodal_ratio=0.70),
                QuestionType.CODE_GENERATION: TypeAllocationConfig(
                    ratio=0.35, multimodal_ratio=0.30
                ),
                QuestionType.FUNCTION_COMPLETION: TypeAllocationConfig(
                    ratio=0.35, multimodal_ratio=0.30
                ),
            }

        return AllocationConfig(
            target_samples=self.gen_config.target_samples,
            type_configs=type_configs,
            over_allocation_factor=self.gen_config.over_allocation_factor,
        )

    def _log_allocation(self, result: AllocationResult) -> None:
        """Log allocation details."""
        console.print(f"    Multimodal: {result.multimodal_samples}")
        console.print(f"    Text-only: {result.text_only_samples}")

        # Allocation metrics
        metrics = result.metrics
        console.print(
            f"    Chunk coverage: {metrics.chunks_used}/{metrics.total_chunks} "
            f"({metrics.chunk_coverage:.1%})"
        )
        console.print(
            f"    Image coverage: {metrics.images_used}/{metrics.total_images} "
            f"({metrics.image_coverage:.1%})"
        )
        if metrics.avg_chunk_usage > 1.0:
            console.print(f"    Avg chunk reuse: {metrics.avg_chunk_usage:.2f}x")

        console.print("    By question type:")
        by_type = result.samples_by_type()
        mm_by_type = result.multimodal_by_type()
        for qt in QuestionType:
            total = by_type[qt]
            mm = mm_by_type[qt]
            if total > 0:
                console.print(f"      {qt.value}: {total} (multimodal: {mm})")

    def _load_checkpoint(self) -> Optional[PlanCheckpoint]:
        """Load checkpoint if exists."""
        if self.no_cache:
            return None

        checkpoint_data = self.checkpoint_manager.load_checkpoint()
        if not checkpoint_data:
            return None

        data, metadata = checkpoint_data
        return PlanCheckpoint(
            tasks=data.get("tasks", []),
            candidates=data.get("candidates", []),
            processed_indices=metadata.get("processed_indices", []),
            phase=metadata.get("phase", "questions"),
        )

    def _save_checkpoint(self, checkpoint: PlanCheckpoint) -> None:
        """Save checkpoint."""
        data = {
            "tasks": checkpoint.tasks,
            "candidates": checkpoint.candidates,
        }
        metadata = {
            "processed_indices": checkpoint.processed_indices,
            "phase": checkpoint.phase,
            "total_tasks": len(checkpoint.tasks),
            "total_candidates": len(checkpoint.candidates),
        }
        self.checkpoint_manager.save_checkpoint(data, metadata)

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
        deprecated_patterns = self._get_deprecated_patterns()

        for pattern, message in deprecated_patterns:
            if re.search(pattern, test_code):
                return f"Deprecated API: {message}"

        return None

    def _get_deprecated_patterns(self) -> list[tuple[str, str]]:
        """Get list of deprecated API patterns for Qiskit 2.0."""
        return [
            (r"from\s+qiskit\.test", "qiskit.test module removed"),
            (r"from\s+qiskit\.opflow", "qiskit.opflow module removed"),
            (r"import\s+qiskit\.opflow", "qiskit.opflow module removed"),
            (r"from\s+qiskit\s+import\s+.*pulse", "qiskit.pulse removed"),
            (r"from\s+qiskit\.pulse", "qiskit.pulse module removed"),
            (r"import\s+qiskit\.pulse", "qiskit.pulse module removed"),
            (
                r"from\s+qiskit\.primitives\s+import\s+.*BaseEstimator",
                "BaseEstimator removed - use StatevectorEstimator",
            ),
            (
                r"from\s+qiskit\.primitives\s+import\s+.*BaseSampler",
                "BaseSampler removed - use StatevectorSampler",
            ),
            (
                r"from\s+qiskit\.primitives\s+import\s+(?!.*Statevector).*\bSampler\b",
                "Sampler removed - use StatevectorSampler",
            ),
            (
                r"from\s+qiskit\.primitives\s+import\s+(?!.*Statevector).*\bEstimator\b",
                "Estimator removed - use StatevectorEstimator",
            ),
            (
                r"from\s+qiskit\.primitives\.utils\s+import\s+.*_circuit_key",
                "_circuit_key is internal API",
            ),
            (
                r"from\s+qiskit\.circuit\s+import\s+.*\bXGate\b",
                "XGate import changed - use from qiskit.circuit.library",
            ),
            (
                r"from\s+qiskit\s+import\s+.*\bParameter\b",
                "Parameter import changed - use from qiskit.circuit",
            ),
            (
                r"from\s+qiskit\.circuit\s+import\s+.*ClassicalBit\b",
                "ClassicalBit removed - use Clbit",
            ),
            (
                r"from\s+qiskit\.providers\.fake_provider\s+import\s+.*FakeVigo",
                "FakeVigo removed - use qiskit_ibm_runtime.fake_provider",
            ),
            (
                r"from\s+qiskit\.circuit\.library\s+import\s+.*random_clifford",
                "random_clifford moved to qiskit.quantum_info",
            ),
            (r"\.bind_parameters\s*\(", "bind_parameters deprecated - use assign_parameters"),
            (r"qiskit\.execute\s*\(", "qiskit.execute removed - use primitives"),
            (r"from\s+qiskit\s+import\s+.*execute", "execute removed - use primitives"),
            (r"\.operation\.qubits", "use circuit_instruction.qubits not .operation.qubits"),
            (r"\.operation\.clbits", "use circuit_instruction.clbits not .operation.clbits"),
            (r"\.index(?!\w)", "Qubit.index removed - use circuit.find_bit()"),
            (r"\.c_if\s*\(", "c_if removed - use if_test or QuantumCircuit.if_test"),
            (r"\.quasi_dists", "quasi_dists removed - use result.data structure"),
            (
                r"StatevectorSampler\s*\([^)]*backend\s*=",
                "StatevectorSampler does not accept backend argument",
            ),
            (
                r"StatevectorEstimator\s*\([^)]*backend\s*=",
                "StatevectorEstimator does not accept backend argument",
            ),
            (r"IBMProvider\s*\(", "IBMProvider removed - use QiskitRuntimeService"),
            (r"from\s+qiskit\s+import\s+.*IBMQ", "IBMQ removed - use QiskitRuntimeService"),
            (r"IBMQ\.load_account", "IBMQ removed - use QiskitRuntimeService"),
            (
                r"excitation_preserving\s*\([^)]*flatten\s*=",
                "excitation_preserving flatten argument removed",
            ),
            (
                r"QuantumCircuit\.\w+\s*\([^)]*condition\s*=",
                "condition argument removed - use if_test",
            ),
            (r"\.true_body\b", "IfElseOp.true_body removed - use .blocks[0]"),
            (r"\.false_body\b", "IfElseOp.false_body removed - use .blocks[1]"),
            (r"Statevector\([^)]*\)\.adjoint(?!\()", "use .adjoint() method not property"),
            (r"\.qasm\s*\(", "QuantumCircuit.qasm() removed - use qasm2.dumps() or qasm3.dumps()"),
            (
                r"ZFeatureMap\s*\(\s*num_qubits\s*=",
                "ZFeatureMap uses positional feature_dimension, not num_qubits kwarg",
            ),
            (
                r"ZZFeatureMap\s*\(\s*num_qubits\s*=",
                "ZZFeatureMap uses positional feature_dimension, not num_qubits kwarg",
            ),
            (
                r"assert\s+.*\s+is\s+.*\.qubits\[",
                "Use == not 'is' for qubit comparison - objects are not identical",
            ),
            (
                r"assert\s+.*\.qubits\[\d+\]\s+is\s+",
                "Use == not 'is' for qubit comparison - objects are not identical",
            ),
            (
                r"from\s+qiskit_machine_learning\.utils\s+import\s+algorithm_globals",
                "algorithm_globals import changed in qiskit_machine_learning",
            ),
        ]

    def _print_summary(
        self, candidates: list[InputCandidate], rejected: int, metrics: AllocationMetrics
    ) -> None:
        """Print summary table."""
        table = Table(title="Planning Complete")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Total Candidates", str(len(candidates)))
        table.add_row("Rejected", str(rejected))
        table.add_row("With Tests", str(sum(1 for c in candidates if c.test_code)))
        table.add_row("Multimodal", str(sum(1 for c in candidates if c.is_multimodal)))
        table.add_row("Text-only", str(sum(1 for c in candidates if not c.is_multimodal)))

        # Allocation metrics
        table.add_row("", "")  # Separator
        table.add_row("Chunks Used", f"{metrics.chunks_used}/{metrics.total_chunks}")
        table.add_row("Chunk Coverage", f"{metrics.chunk_coverage:.1%}")
        table.add_row("Images Used", f"{metrics.images_used}/{metrics.total_images}")
        table.add_row("Image Coverage", f"{metrics.image_coverage:.1%}")
        if metrics.avg_chunk_usage > 1.0:
            table.add_row("Avg Chunk Reuse", f"{metrics.avg_chunk_usage:.2f}x")

        console.print("\n")
        console.print(table)

        # Type breakdown
        type_table = Table(title="By Question Type")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Count", style="green", justify="right")
        type_table.add_column("Multimodal", style="yellow", justify="right")

        by_type = {}
        mm_by_type = {}
        for c in candidates:
            qt = c.question_type
            by_type[qt] = by_type.get(qt, 0) + 1
            if c.is_multimodal:
                mm_by_type[qt] = mm_by_type.get(qt, 0) + 1

        for qt in QuestionType:
            if qt in by_type:
                type_table.add_row(qt.value, str(by_type[qt]), str(mm_by_type.get(qt, 0)))

        console.print("\n")
        console.print(type_table)

        console.print(f"\n[green]✓ Saved to: {self.output_file}[/green]")
