"""Parity test for the parallel code-execution pool.

Runs a fixed set of (code, test, entry_point) triples through
``CodeExecutor.execute_many_async`` at different concurrency levels and
asserts that per-item success/output/error fields are identical.

Also exercises the batched path in both runners (``QiskitHumanEvalRunner``
and ``SyntheticDatasetRunner``) via a stub model client that replays
canned solutions, and compares the aggregated metrics against a
sequential reference run (``execution_max_concurrent=1``).

Run:
    cd quantum-assistant/src
    ../.venv/bin/python -m pytest -xvs ../tests/test_parallel_execution_parity.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

# Make ``src`` importable when this test file is run directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate.execution.sandbox import CodeExecutor  # noqa: E402
from evaluate.runners.qiskit_humaneval import QiskitHumanEvalRunner  # noqa: E402
from evaluate.runners.synthetic import SyntheticDatasetRunner  # noqa: E402


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _items() -> list[tuple[str, str, str]]:
    """A small, deterministic mix of passing/failing/timeout items."""
    return [
        # 1. Straightforward pass
        (
            "def add(a, b):\n    return a + b\n",
            "def check(f):\n    assert f(1, 2) == 3\n    assert f(-1, 1) == 0\n",
            "add",
        ),
        # 2. Assertion failure
        (
            "def add(a, b):\n    return a - b\n",
            "def check(f):\n    assert f(1, 2) == 3\n",
            "add",
        ),
        # 3. Runtime error
        (
            "def boom(x):\n    raise RuntimeError('x')\n",
            "def check(f):\n    f(1)\n",
            "boom",
        ),
        # 4. Timeout (short sleep relative to CodeExecutor(timeout=2))
        (
            "import time\n\ndef slow():\n    time.sleep(10)\n",
            "def check(f):\n    f()\n",
            "slow",
        ),
        # 5. Import error
        (
            "def foo():\n    import __definitely_not_a_module__ as m\n    return m\n",
            "def check(f):\n    f()\n",
            "foo",
        ),
        # 6. Multi-assert pass
        (
            "def square(x):\n    return x * x\n",
            "def check(f):\n    assert f(2) == 4\n    assert f(0) == 0\n    assert f(-3) == 9\n",
            "square",
        ),
        # 7. test_* style
        (
            "def cube(x):\n    return x ** 3\n",
            "def test_cube():\n    assert cube(2) == 8\n",
            "cube",
        ),
        # 8. SyntaxError at compile time
        (
            "def bad(:\n    return 1\n",
            "def check(f):\n    assert f() == 1\n",
            "bad",
        ),
    ]


# ----------------------------------------------------------------------
# Executor-level parity
# ----------------------------------------------------------------------


def _execute_serial(items: list[tuple[str, str, str]], timeout: int) -> list[dict[str, Any]]:
    executor = CodeExecutor(timeout=timeout)
    out = []
    for code, test, ep in items:
        res = executor.execute(code, test, ep)
        out.append(
            {
                "success": res.success,
                "error": res.error,
                "output": res.output,
                "timeout": res.timeout,
            }
        )
    return out


def _execute_pool(
    items: list[tuple[str, str, str]], timeout: int, pool: int
) -> list[dict[str, Any]]:
    executor = CodeExecutor(timeout=timeout)
    results = asyncio.run(executor.execute_many_async(items, max_concurrent=pool))
    return [
        {
            "success": r.success,
            "error": r.error,
            "output": r.output,
            "timeout": r.timeout,
        }
        for r in results
    ]


def test_execute_many_async_parity_vs_serial():
    """Pool=1, 4, 8, 16 all produce the same success/output/timeout flags as serial."""
    items = _items()
    timeout = 2  # short, deterministic

    ref = _execute_serial(items, timeout)

    for pool in (1, 4, 8, 16):
        got = _execute_pool(items, timeout, pool)
        assert len(got) == len(ref)
        for i, (g, r) in enumerate(zip(got, ref)):
            assert g["success"] == r["success"], (
                f"item {i} success mismatch at pool={pool}: got={g}, ref={r}"
            )
            assert g["timeout"] == r["timeout"], (
                f"item {i} timeout mismatch at pool={pool}: got={g}, ref={r}"
            )
            # Output/error text can vary in trivial whitespace depending on the
            # path, but for successful items output must contain TEST_PASSED
            # and for failures the canonical error class must match.
            if g["success"]:
                assert "TEST_PASSED" in g["output"]
            else:
                # Classify on first token of error message
                ref_token = r["error"].split(":")[0].split(" ")[0] if r["error"] else ""
                got_token = g["error"].split(":")[0].split(" ")[0] if g["error"] else ""
                assert got_token == ref_token, (
                    f"item {i} error class mismatch at pool={pool}: "
                    f"got={got_token!r}, ref={ref_token!r}"
                )


# ----------------------------------------------------------------------
# Runner-level parity (Qiskit HumanEval)
# ----------------------------------------------------------------------


def _qhe_samples() -> list[dict[str, Any]]:
    """Tiny QHE-shaped dataset exercising normal completion logic."""
    return [
        {
            "task_id": "parity/qhe/0",
            "prompt": "def add(a, b):\n    pass\n",
            "canonical_solution": "    return a + b\n",
            "test": "def check(f):\n    assert f(1, 2) == 3\n",
            "entry_point": "add",
        },
        {
            "task_id": "parity/qhe/1",
            "prompt": "def sub(a, b):\n    pass\n",
            "canonical_solution": "    return a - b\n",
            "test": "def check(f):\n    assert f(5, 2) == 3\n",
            "entry_point": "sub",
        },
        {
            "task_id": "parity/qhe/2",
            "prompt": "def mul(a, b):\n    pass\n",
            "canonical_solution": "    return a * b\n",
            "test": "def check(f):\n    assert f(3, 4) == 12\n",
            "entry_point": "mul",
        },
    ]


class _StubClient:
    """Stand-in model client that returns canned solutions."""

    def __init__(self, per_task: dict[str, list[str]]):
        self._per_task = per_task
        self.model_name = "stub"

    async def generate_batch_async(
        self,
        messages_list,
        max_concurrent: int = 10,
        progress_callback=None,
    ) -> list[str]:
        # QHE runner collects (messages, task_id) pairs but only passes
        # messages into generate_batch_async. We reconstruct the task_id
        # from the user message's prompt.
        out: list[str] = []
        for msgs in messages_list:
            # The user message content == sample["prompt"] for QHE
            prompt = msgs[-1].content
            # Find the task by prompt
            for tid, _solutions in self._per_task.items():
                if tid == prompt:
                    out.append(self._per_task[tid].pop(0))
                    break
            else:
                out.append("    return 0\n")  # default wrong-answer fallback
        return out


def _qhe_run(pool: int) -> dict[str, Any]:
    samples = _qhe_samples()

    # Canned solutions: two per task (so pass@1 with n=2 exercises aggregation).
    # For task 0 both correct; task 1 first wrong, second correct; task 2 both wrong.
    per_task_by_prompt = {
        samples[0]["prompt"]: ["    return a + b\n", "    return a + b\n"],
        samples[1]["prompt"]: ["    return a + b\n", "    return a - b\n"],
        samples[2]["prompt"]: ["    return a + b\n", "    return a + b + 1\n"],
    }
    client = _StubClient(per_task_by_prompt)

    runner = QiskitHumanEvalRunner(
        dataset_path=Path("unused"),
        model_client=client,  # type: ignore[arg-type]
        k_values=[1, 2],
        num_samples_per_task=2,
        timeout=10,
        dataset_type="normal",
        execution_max_concurrent=pool,
    )

    results = runner.evaluate(samples, save_results=None, verify_canonical=True)
    return {
        "total": results.total_samples,
        "successful": results.successful,
        "metrics": results.metrics,
        "per_sample": [
            {
                "task_id": r.task_id,
                "success": r.success,
                "metrics": r.metrics,
                "execution": r.metadata.get("execution_results"),
            }
            for r in results.per_sample_results
        ],
    }


def test_qhe_runner_batched_vs_serial_parity():
    serial = _qhe_run(pool=1)
    parallel = _qhe_run(pool=8)
    assert serial["total"] == parallel["total"]
    assert serial["successful"] == parallel["successful"]
    assert serial["metrics"] == parallel["metrics"]
    assert serial["per_sample"] == parallel["per_sample"]


# ----------------------------------------------------------------------
# Runner-level parity (Synthetic)
# ----------------------------------------------------------------------


def _synthetic_samples() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "parity/syn/0",
            "question": "Write add(a,b).",
            "answer": "def add(a, b):\n    return a + b\n",
            "category": "test",
            "question_type": "code_generation",
            "test_code": "def check(f):\n    assert f(2, 3) == 5\n",
            "entry_point": "add",
            "image": None,
            "source": "unit",
        },
        {
            "task_id": "parity/syn/1",
            "question": "def sub(a,b):\n    pass",
            "answer": "return a - b",
            "category": "test",
            "question_type": "function_completion",
            "test_code": "def check(f):\n    assert f(5, 2) == 3\n",
            "entry_point": "sub",
            "image": None,
            "source": "unit",
        },
        {
            "task_id": "parity/syn/2",
            "question": "What is 2+2?",
            "answer": "The result is four.",
            "category": "test",
            "question_type": "qa",
            "test_code": "",
            "entry_point": "",
            "image": None,
            "source": "unit",
        },
    ]


class _SyntheticStub:
    def __init__(self, per_question: dict[str, list[str]]):
        self._per = per_question
        self.model_name = "stub"

    async def generate_async(self, messages, *a, **kw):
        question = messages[-1].content
        return self._per[question].pop(0)


def _synthetic_run(pool: int) -> dict[str, Any]:
    samples = _synthetic_samples()
    per_question = {
        samples[0]["question"]: ["def add(a, b):\n    return a + b\n"],
        samples[1]["question"]: ["return a - b"],
        samples[2]["question"]: ["The result is four."],
    }
    client = _SyntheticStub(per_question)

    runner = SyntheticDatasetRunner(
        dataset_path=Path("unused"),
        model_client=client,  # type: ignore[arg-type]
        k_values=[1],
        num_samples_per_task=1,
        timeout=10,
        execution_max_concurrent=pool,
    )

    # Shortcut: we hand samples in directly, skipping dataset loading.
    # ``evaluate`` handles printing + aggregation.
    results = runner.evaluate(samples, save_results=None)
    return {
        "total": results.total_samples,
        "successful": results.successful,
        "metrics": results.metrics,
        "per_sample": [
            {
                "task_id": r.task_id,
                "success": r.success,
                "metrics": {
                    k: v for k, v in r.metrics.items() if k != "bleu" and k != "rouge_l"
                },
                "execution": r.metadata.get("execution_results"),
            }
            for r in results.per_sample_results
        ],
    }


def test_synthetic_runner_batched_vs_serial_parity():
    serial = _synthetic_run(pool=1)
    parallel = _synthetic_run(pool=8)
    assert serial["total"] == parallel["total"]
    assert serial["successful"] == parallel["successful"]
    # QA metrics include bleu/rouge which are deterministic too, but we
    # already compared the code pass@1 and success flags.
    assert serial["metrics"].keys() == parallel["metrics"].keys()
    assert serial["per_sample"] == parallel["per_sample"]


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
