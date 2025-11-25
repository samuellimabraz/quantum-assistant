"""Tests for evaluation module."""

import json
import tempfile
from pathlib import Path

import pytest

from evaluate.evaluators.base import EvaluationResult
from evaluate.evaluators.code import CodeEvaluator
from evaluate.evaluators.multimodal import MultimodalEvaluator
from evaluate.execution.sandbox import CodeExecutor
from evaluate.metrics.code_metrics import PassAtK
from evaluate.metrics.text_metrics import BLEU, ExactMatch, ROUGE


class TestCodeExecutor:
    """Tests for CodeExecutor."""

    def test_successful_execution(self):
        """Test successful code execution."""
        executor = CodeExecutor(timeout=5)

        code = """
def add(a, b):
    return a + b
"""
        test_code = """
assert add(2, 3) == 5
assert add(-1, 1) == 0
"""

        result = executor.execute(code, test_code, "add")

        assert result.success
        assert not result.timeout
        assert result.error == ""

    def test_execution_failure(self):
        """Test code execution failure."""
        executor = CodeExecutor(timeout=5)

        code = """
def add(a, b):
    return a - b  # Wrong implementation
"""
        test_code = """
assert add(2, 3) == 5
"""

        result = executor.execute(code, test_code, "add")

        assert not result.success
        assert "assertion" in result.error.lower()

    def test_execution_timeout(self):
        """Test code execution timeout."""
        executor = CodeExecutor(timeout=1)

        code = """
import time
def slow_function():
    time.sleep(10)
    return 42
"""
        test_code = """
result = slow_function()
assert result == 42
"""

        result = executor.execute(code, test_code, "slow_function")

        assert not result.success
        assert result.timeout

    def test_syntax_error(self):
        """Test handling of syntax errors."""
        executor = CodeExecutor(timeout=5)

        code = """
def broken(
    return 42
"""

        result = executor.execute(code, "", "broken")

        assert not result.success
        assert "SyntaxError" in result.error or "syntax" in result.error.lower()


class TestPassAtK:
    """Tests for Pass@k metric."""

    def test_pass_at_1(self):
        """Test Pass@1 metric."""
        metric = PassAtK(k=1)

        # All fail
        assert metric.compute([False, False, False]) == 0.0

        # At least one passes
        assert metric.compute([True, False, False]) > 0.0
        assert metric.compute([False, True, False]) > 0.0

        # All pass
        assert metric.compute([True, True, True]) == 1.0

    def test_pass_at_k_computation(self):
        """Test Pass@k computation with different values."""
        # Example from HumanEval paper
        n, c = 10, 3

        pass_at_1 = PassAtK.compute_pass_at_k(n, c, 1)
        pass_at_5 = PassAtK.compute_pass_at_k(n, c, 5)
        pass_at_10 = PassAtK.compute_pass_at_k(n, c, 10)

        # pass@k should increase with k
        assert 0.0 <= pass_at_1 <= pass_at_5 <= pass_at_10 <= 1.0

    def test_edge_cases(self):
        """Test edge cases for Pass@k."""
        # Empty list
        assert PassAtK(k=1).compute([]) == 0.0

        # n < k
        assert PassAtK(k=5).compute([True, False]) == 1.0
        assert PassAtK(k=5).compute([False, False]) == 0.0


class TestTextMetrics:
    """Tests for text-based metrics."""

    def test_exact_match(self):
        """Test exact match metric."""
        metric = ExactMatch()

        # Perfect match
        assert metric.compute(["hello"], ["hello"]) == 1.0

        # No match
        assert metric.compute(["hello"], ["world"]) == 0.0

        # Partial match (multiple samples)
        assert metric.compute(["hello", "world"], ["hello", "earth"]) == 0.5

    def test_bleu_score(self):
        """Test BLEU score."""
        metric = BLEU(n=4)

        # Identical texts
        assert metric.compute(["hello world"], ["hello world"]) > 0.9

        # Completely different
        score = metric.compute(["apple banana"], ["car truck"])
        assert 0.0 <= score < 0.1

    def test_rouge_l_score(self):
        """Test ROUGE-L score."""
        metric = ROUGE()

        # Identical texts
        assert metric.compute(["hello world"], ["hello world"]) == 1.0

        # Similar texts
        score = metric.compute(["hello world program"], ["hello program"])
        assert 0.5 < score < 1.0


class TestCodeEvaluator:
    """Tests for CodeEvaluator."""

    def test_evaluate_simple_sample(self):
        """Test evaluating a simple code sample."""
        evaluator = CodeEvaluator(k_values=[1], timeout=5)

        sample = {
            "task_id": "test/0",
            "test": """
assert create_list(3) == [0, 1, 2]
""",
            "entry_point": "create_list",
        }

        # Correct solution
        predictions = [
            """
def create_list(n):
    return list(range(n))
"""
        ]

        result = evaluator.evaluate_sample(sample, predictions)

        assert result.success
        assert result.task_id == "test/0"
        assert "pass@1" in result.metrics
        assert result.metrics["pass@1"] == 1.0

    def test_evaluate_multiple_predictions(self):
        """Test evaluating multiple predictions."""
        evaluator = CodeEvaluator(k_values=[1, 2], timeout=5)

        sample = {
            "task_id": "test/1",
            "test": """
assert double(5) == 10
""",
            "entry_point": "double",
        }

        # One correct, one incorrect
        predictions = [
            "def double(x):\n    return x * 2",
            "def double(x):\n    return x + 2",
        ]

        result = evaluator.evaluate_sample(sample, predictions)

        assert result.success  # At least one passes
        assert result.metadata["num_passed"] == 1

    def test_aggregate_results(self):
        """Test result aggregation."""
        evaluator = CodeEvaluator(k_values=[1], timeout=5)

        results = [
            EvaluationResult(
                task_id="test/0",
                success=True,
                predictions=["code"],
                metrics={"pass@1": 1.0, "pass_rate": 1.0},
                metadata={"execution_results": [True], "num_passed": 1},
            ),
            EvaluationResult(
                task_id="test/1",
                success=False,
                predictions=["code"],
                metrics={"pass@1": 0.0, "pass_rate": 0.0},
                metadata={"execution_results": [False], "num_passed": 0},
            ),
        ]

        aggregated = evaluator.aggregate_results(results)

        assert aggregated.total_samples == 2
        assert aggregated.successful == 1
        assert aggregated.failed == 1
        assert aggregated.success_rate == 0.5


class TestMultimodalEvaluator:
    """Tests for MultimodalEvaluator."""

    def test_evaluate_qa_sample(self):
        """Test evaluating Q&A sample."""
        evaluator = MultimodalEvaluator(timeout=5)

        sample = {
            "task_id": "qa/0",
            "question_type": "qa",
            "question": "What is a qubit?",
            "answer": "A qubit is a quantum bit, the basic unit of quantum information.",
        }

        predictions = ["A qubit is a quantum bit, which is the basic unit of quantum information."]

        result = evaluator.evaluate_sample(sample, predictions)

        assert "rouge_l" in result.metrics
        assert "bleu" in result.metrics
        assert result.metrics["rouge_l"] > 0.5

    def test_evaluate_code_sample(self):
        """Test evaluating code sample."""
        evaluator = MultimodalEvaluator(timeout=5)

        sample = {
            "task_id": "code/0",
            "question_type": "code",
            "question": "Create a quantum circuit",
            "answer": "qc = QuantumCircuit(2)",
            "test": """
from qiskit import QuantumCircuit
assert isinstance(create_circuit(), QuantumCircuit)
""",
            "entry_point": "create_circuit",
        }

        predictions = [
            """
from qiskit import QuantumCircuit
def create_circuit():
    return QuantumCircuit(2)
"""
        ]

        result = evaluator.evaluate_sample(sample, predictions)

        assert result.success
        assert "pass@1" in result.metrics

    def test_aggregate_by_type(self):
        """Test aggregation by question type."""
        evaluator = MultimodalEvaluator(timeout=5)

        results = [
            EvaluationResult(
                task_id="qa/0",
                success=True,
                predictions=["answer"],
                metrics={"rouge_l": 0.8},
                metadata={"question_type": "qa"},
            ),
            EvaluationResult(
                task_id="code/0",
                success=True,
                predictions=["code"],
                metrics={"pass@1": 1.0},
                metadata={"question_type": "code"},
            ),
            EvaluationResult(
                task_id="qa/1",
                success=False,
                predictions=["answer"],
                metrics={"rouge_l": 0.2},
                metadata={"question_type": "qa"},
            ),
        ]

        aggregated = evaluator.aggregate_results(results)

        assert "by_type.qa" in aggregated.metrics
        assert "by_type.code" in aggregated.metrics
        assert aggregated.metrics["by_type.qa"]["count"] == 2
        assert aggregated.metrics["by_type.code"]["count"] == 1


class TestQiskitHumanEvalIntegration:
    """Integration tests for Qiskit HumanEval evaluation."""

    def test_load_dataset(self):
        """Test loading Qiskit HumanEval dataset."""
        # Create a minimal test dataset
        test_data = [
            {
                "task_id": "test/0",
                "prompt": "def add(a, b):\n    pass",
                "canonical_solution": "    return a + b",
                "test": "assert add(2, 3) == 5",
                "entry_point": "add",
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            # Load and verify
            with open(temp_path, encoding="utf-8") as f:
                loaded = json.load(f)

            assert len(loaded) == 1
            assert loaded[0]["task_id"] == "test/0"

        finally:
            temp_path.unlink()


class TestQiskitHumanEvalRunner:
    """Tests for QiskitHumanEvalRunner."""

    def test_combine_code_normal_dataset(self):
        """Test code combination for normal (completion) dataset."""
        from evaluate.runners.qiskit_humaneval import QiskitHumanEvalRunner, DatasetType
        from models.client import LLMClient

        # Create dummy client
        client = LLMClient(base_url="http://localhost", model_name="dummy")

        # Create test dataset file
        test_data = [
            {
                "task_id": "test/0",
                "prompt": 'def add(a, b):\n    """Add two numbers."""\n',
                "canonical_solution": "    return a + b\n",
                "test": "assert add(2, 3) == 5",
                "entry_point": "add",
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            runner = QiskitHumanEvalRunner(
                dataset_path=temp_path,
                model_client=client,
                dataset_type=DatasetType.NORMAL,
            )

            sample = test_data[0]

            # Test combining with generated code
            generated = "    return a + b\n"
            combined = runner.combine_code(sample, generated)

            # Should be prompt + generated (trailing newlines stripped)
            assert "def add(a, b):" in combined
            assert "return a + b" in combined

            # Test combining with canonical
            canonical_combined = runner.combine_canonical(sample)
            assert "def add(a, b):" in canonical_combined
            assert "return a + b" in canonical_combined

        finally:
            temp_path.unlink()

    def test_combine_code_hard_dataset(self):
        """Test code combination for hard (full generation) dataset."""
        from evaluate.runners.qiskit_humaneval import QiskitHumanEvalRunner, DatasetType
        from models.client import LLMClient

        client = LLMClient(base_url="http://localhost", model_name="dummy")

        test_data = [
            {
                "task_id": "test/0",
                "prompt": "Create a function that adds two numbers.",
                "canonical_solution": "def add(a, b):\n    return a + b\n",
                "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n\ncheck(add)",
                "entry_point": "add",
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix="_hard.json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            runner = QiskitHumanEvalRunner(
                dataset_path=temp_path,
                model_client=client,
                dataset_type=DatasetType.HARD,
            )

            sample = test_data[0]

            # For hard dataset, generated code should be used as-is (minus trailing newlines)
            generated = "def add(a, b):\n    return a + b\n"
            combined = runner.combine_code(sample, generated)
            # Prompt should NOT be in the combined code for hard dataset
            assert "Create a function" not in combined
            assert "def add(a, b):" in combined
            assert "return a + b" in combined

            # Canonical should also be used as-is
            canonical_combined = runner.combine_canonical(sample)
            assert "def add(a, b):" in canonical_combined
            assert "return a + b" in canonical_combined

        finally:
            temp_path.unlink()

    def test_canonical_verification(self):
        """Test that canonical solutions pass their tests."""
        from evaluate.runners.qiskit_humaneval import QiskitHumanEvalRunner, DatasetType
        from models.client import LLMClient

        client = LLMClient(base_url="http://localhost", model_name="dummy")

        # Create test with valid canonical solution
        test_data = [
            {
                "task_id": "test/0",
                "prompt": 'def add(a, b):\n    """Add two numbers."""\n',
                "canonical_solution": "    return a + b\n",
                "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
                "entry_point": "add",
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            runner = QiskitHumanEvalRunner(
                dataset_path=temp_path,
                model_client=client,
                dataset_type=DatasetType.NORMAL,
            )

            samples = runner.load_dataset()
            result = runner.verify_canonical_solutions(samples)

            assert result["passed"] == 1
            assert result["failed"] == 0

        finally:
            temp_path.unlink()

    def test_extract_code_from_markdown(self):
        """Test extracting code from markdown code blocks."""
        from evaluate.runners.qiskit_humaneval import QiskitHumanEvalRunner
        from models.client import LLMClient

        client = LLMClient(base_url="http://localhost", model_name="dummy")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"task_id": "test"}], f)
            temp_path = Path(f.name)

        try:
            runner = QiskitHumanEvalRunner(
                dataset_path=temp_path,
                model_client=client,
            )

            # Test markdown code block extraction - preserves indentation
            response = "```python\n    return a + b\n```"
            extracted = runner.extract_code_from_response(response)
            assert extracted == "    return a + b"

            # Test without code block - preserves indentation
            response2 = "    return a + b"
            extracted2 = runner.extract_code_from_response(response2)
            assert extracted2 == "    return a + b"

        finally:
            temp_path.unlink()

    def test_extract_code_multiple_blocks(self):
        """Test extracting correct code block when response has multiple blocks."""
        from evaluate.runners.qiskit_humaneval import QiskitHumanEvalRunner
        from models.client import LLMClient

        client = LLMClient(base_url="http://localhost", model_name="dummy")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"task_id": "test"}], f)
            temp_path = Path(f.name)

        try:
            runner = QiskitHumanEvalRunner(
                dataset_path=temp_path,
                model_client=client,
            )

            # Response with example call first, then full implementation
            response = """Here's the solution:

```python
my_function()
```

The full implementation:

```python
def my_function():
    return 42
```
"""
            # Should find the block with the function definition
            extracted = runner.extract_code_from_response(response, entry_point="my_function")
            assert "def my_function():" in extracted
            assert "return 42" in extracted
            assert extracted.strip() != "my_function()"

            # Without entry_point, should return largest block
            extracted_no_entry = runner.extract_code_from_response(response)
            assert "def my_function():" in extracted_no_entry

        finally:
            temp_path.unlink()

    def test_combine_code_normal_with_full_code_generation(self):
        """Test that normal dataset handles model generating full code instead of body."""
        from evaluate.runners.qiskit_humaneval import QiskitHumanEvalRunner, DatasetType
        from models.client import LLMClient

        client = LLMClient(base_url="http://localhost", model_name="dummy")

        test_data = [
            {
                "task_id": "test/0",
                "prompt": 'from qiskit import QuantumCircuit\ndef create_circuit(n):\n    """Create a circuit."""\n',
                "canonical_solution": "    return QuantumCircuit(n)\n",
                "test": "assert create_circuit(3).num_qubits == 3",
                "entry_point": "create_circuit",
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            runner = QiskitHumanEvalRunner(
                dataset_path=temp_path,
                model_client=client,
                dataset_type=DatasetType.NORMAL,
            )

            sample = test_data[0]

            # Model generates full code with function definition (common with instruction-tuned models)
            generated_full = """```python
from qiskit import QuantumCircuit

def create_circuit(n):
    return QuantumCircuit(n)
```"""
            combined = runner.combine_code(sample, generated_full)

            # Should use generated code directly (not concatenate with prompt)
            # because it contains the function definition
            assert combined.count("def create_circuit") == 1
            assert "SyntaxError" not in combined  # No invalid concatenation
            assert "return QuantumCircuit(n)" in combined

            # Model generates just the function body (expected behavior)
            generated_body = "    return QuantumCircuit(n)\n"
            combined_body = runner.combine_code(sample, generated_body)

            # Should concatenate with prompt
            assert "from qiskit import QuantumCircuit" in combined_body
            assert "def create_circuit(n):" in combined_body
            assert "return QuantumCircuit(n)" in combined_body

        finally:
            temp_path.unlink()

    def test_dataset_type_detection(self):
        """Test auto-detection of dataset type from filename."""
        from evaluate.runners.qiskit_humaneval import QiskitHumanEvalRunner, DatasetType
        from models.client import LLMClient

        client = LLMClient(base_url="http://localhost", model_name="dummy")

        # Test normal dataset detection
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            normal_path = Path(f.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_hard.json", delete=False, prefix="dataset"
        ) as f:
            json.dump([], f)
            hard_path = Path(f.name)

        try:
            runner_normal = QiskitHumanEvalRunner(
                dataset_path=normal_path,
                model_client=client,
            )
            assert runner_normal.dataset_type == DatasetType.NORMAL

            runner_hard = QiskitHumanEvalRunner(
                dataset_path=hard_path,
                model_client=client,
            )
            assert runner_hard.dataset_type == DatasetType.HARD

        finally:
            normal_path.unlink()
            hard_path.unlink()

    def test_create_messages_with_system_prompt(self):
        """Test that system prompt is sent as system message."""
        from evaluate.runners.qiskit_humaneval import QiskitHumanEvalRunner
        from models.client import LLMClient, Message

        client = LLMClient(base_url="http://localhost", model_name="test")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"task_id": "test/0", "prompt": "Write a function"}], f)
            dataset_path = Path(f.name)

        try:
            runner = QiskitHumanEvalRunner(
                dataset_path=dataset_path,
                model_client=client,
            )

            sample = {"task_id": "test/0", "prompt": "Write a function"}

            # Test without system prompt
            messages_no_sys = runner.create_messages(sample, system_prompt=None)
            assert len(messages_no_sys) == 1
            assert messages_no_sys[0].role == "user"
            assert messages_no_sys[0].content == "Write a function"

            # Test with system prompt
            messages_with_sys = runner.create_messages(
                sample, system_prompt="You are a helpful assistant."
            )
            assert len(messages_with_sys) == 2
            assert messages_with_sys[0].role == "system"
            assert messages_with_sys[0].content == "You are a helpful assistant."
            assert messages_with_sys[1].role == "user"
            assert messages_with_sys[1].content == "Write a function"

        finally:
            dataset_path.unlink()

    def test_system_prompts(self):
        """Test that system prompts are retrieved correctly."""
        from evaluate.config.system_prompts import get_system_prompt, DEFAULT_PROMPTS

        # Test predefined prompts
        qiskit_prompt = get_system_prompt("qiskit_humaneval")
        assert qiskit_prompt is not None
        assert "Qiskit code assistant" in qiskit_prompt
        assert "2.0" in qiskit_prompt  # Qiskit version mentioned
        assert "transpile" in qiskit_prompt.lower()  # Mentions deprecated methods

        minimal_prompt = get_system_prompt("qiskit_humaneval_minimal")
        assert minimal_prompt is not None
        assert len(minimal_prompt) < len(qiskit_prompt)  # Minimal should be shorter

        generic_prompt = get_system_prompt("generic")
        assert generic_prompt is not None
        assert "Qiskit" not in generic_prompt  # Generic should not mention Qiskit

        # Test custom prompt
        custom = "Custom prompt text"
        assert get_system_prompt("qiskit_humaneval", custom_prompt=custom) == custom

        # Test default
        default = get_system_prompt()
        assert default == DEFAULT_PROMPTS["qiskit_humaneval"]


class TestResultsManager:
    """Tests for ResultsManager."""

    def test_sanitize_model_name(self):
        """Test model name sanitization."""
        from evaluate.utils.results import ResultsManager

        # Test basic sanitization
        assert ResultsManager.sanitize_model_name("gpt-4") == "gpt-4"
        assert ResultsManager.sanitize_model_name("qwen2.5-coder-14b") == "qwen2.5-coder-14b"

        # Test slash replacement
        assert ResultsManager.sanitize_model_name("openai/gpt-4") == "openai-gpt-4"
        assert ResultsManager.sanitize_model_name("Qiskit/Qwen2.5-Coder") == "qiskit-qwen2.5-coder"

        # Test special characters
        assert ResultsManager.sanitize_model_name("model name") == "model-name"
        assert ResultsManager.sanitize_model_name("model@v1") == "model-v1"

    def test_generate_filename(self):
        """Test filename generation."""
        from datetime import datetime
        from evaluate.utils.results import ResultsManager

        timestamp = datetime(2024, 11, 25, 14, 30, 22)

        filename = ResultsManager.generate_filename(
            model_name="qwen2.5-coder-14b",
            num_samples_per_task=10,
            k_values=[1, 5, 10],
            timestamp=timestamp,
        )

        assert filename == "qwen2.5-coder-14b_n10_k1-5-10_20241125_143022.json"

    def test_generate_filename_with_suffix(self):
        """Test filename generation with suffix."""
        from datetime import datetime
        from evaluate.utils.results import ResultsManager

        timestamp = datetime(2024, 11, 25, 14, 30, 22)

        filename = ResultsManager.generate_filename(
            model_name="gpt-4",
            num_samples_per_task=5,
            k_values=[1],
            timestamp=timestamp,
            suffix="test",
        )

        assert filename == "gpt-4_n5_k1_20241125_143022_test.json"

    def test_get_dataset_dir(self):
        """Test dataset directory selection."""
        from evaluate.utils.results import ResultsManager

        # Normal qiskit humaneval
        path = ResultsManager.get_dataset_dir("qiskit_humaneval", None)
        assert path == Path("outputs/evaluate/qiskit-humaneval")

        path = ResultsManager.get_dataset_dir("qiskit_humaneval", "normal")
        assert path == Path("outputs/evaluate/qiskit-humaneval")

        # Hard qiskit humaneval
        path = ResultsManager.get_dataset_dir("qiskit_humaneval", "hard")
        assert path == Path("outputs/evaluate/qiskit-humaneval-hard")

        # Synthetic
        path = ResultsManager.get_dataset_dir("synthetic", None)
        assert path == Path("outputs/evaluate/synthetic")

    def test_get_result_path(self):
        """Test full result path generation."""
        from datetime import datetime
        from evaluate.utils.results import ResultsManager

        timestamp = datetime(2024, 11, 25, 14, 30, 22)

        path = ResultsManager.get_result_path(
            dataset_type="qiskit_humaneval",
            model_name="openai/gpt-oss-120b",
            num_samples_per_task=10,
            k_values=[1, 5, 10],
            dataset_variant="hard",
            timestamp=timestamp,
        )

        expected = Path(
            "outputs/evaluate/qiskit-humaneval-hard/"
            "openai-gpt-oss-120b_n10_k1-5-10_20241125_143022.json"
        )
        assert path == expected

    def test_build_metadata(self):
        """Test metadata building."""
        from datetime import datetime
        from evaluate.utils.results import ResultsManager

        timestamp = datetime(2024, 11, 25, 14, 30, 22)

        metadata = ResultsManager.build_metadata(
            dataset_path="/path/to/dataset.json",
            dataset_type="qiskit_humaneval",
            dataset_variant="normal",
            model_name="gpt-4",
            num_samples=100,
            num_samples_per_task=10,
            k_values=[1, 5, 10],
            timeout=30,
            system_prompt="Test system prompt",
            verify_canonical=True,
            timestamp=timestamp,
        )

        assert metadata["run_info"]["timestamp"] == "2024-11-25T14:30:22"
        assert metadata["model"]["name"] == "gpt-4"
        assert metadata["dataset"]["type"] == "qiskit_humaneval"
        assert metadata["dataset"]["variant"] == "normal"
        assert metadata["dataset"]["num_samples"] == 100
        assert metadata["evaluation"]["solutions_per_task"] == 10
        assert metadata["evaluation"]["k_values"] == [1, 5, 10]
        assert metadata["evaluation"]["system_prompt"] == "Test system prompt"
        assert metadata["evaluation"]["verify_canonical"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
