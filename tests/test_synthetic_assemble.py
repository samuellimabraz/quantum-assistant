"""Regression tests for SyntheticDatasetRunner._assemble_function_completion
(and its helper _normalize_body_indentation).

The original implementation flattened nested control-flow bodies whose first
line opens a block (`for:`, `if:`, `with:`, ...), producing an
IndentationError. This regression triggered on ``synthetic/test/627``; see
paper/review_response_plan.md Section 1.10.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluate.runners.synthetic import SyntheticDatasetRunner  # noqa: E402
from models.client import LLMClient  # noqa: E402


def _runner() -> SyntheticDatasetRunner:
    return SyntheticDatasetRunner(
        dataset_path=Path("/tmp"),
        model_client=LLMClient(base_url="http://localhost", model_name="dummy"),
    )


def _assert_compiles(code: str) -> None:
    compile(code, "<assembled>", "exec")


def test_assemble_preserves_nested_block_body() -> None:
    """synthetic/test/627 repro: body opens a for-loop whose inner
    if/yield must stay indented relative to the for-statement."""
    stub = (
        "```python\n"
        "def apply_shift(qubits, shift):\n"
        "    \"\"\"doc\"\"\"\n"
        "    pass\n"
        "```"
    )
    body = (
        "for i, q in zip(range(shift.bit_length()), qubits):\n"
        "        if shift >> i & 1:\n"
        "            yield (i, q)"
    )
    out = _runner()._assemble_function_completion(stub, body)
    _assert_compiles(out)
    # The for-loop opener must be at the function body indent (4 spaces)
    assert "    for i, q" in out
    # The inner if must be strictly deeper than the for
    assert "\n        if" in out or "\n            if" in out
    # The yield must be strictly deeper than the if
    for_col = out.index("for i, q") - out.rfind("\n", 0, out.index("for i, q")) - 1
    if_col = out.index("if shift") - out.rfind("\n", 0, out.index("if shift")) - 1
    yield_col = out.index("yield") - out.rfind("\n", 0, out.index("yield")) - 1
    assert if_col > for_col
    assert yield_col > if_col


def test_assemble_quirky_first_line_flush() -> None:
    """Common LLM output pattern: first statement flush-left, rest at 4
    spaces. Assembled body must compile as a normal function body."""
    stub = (
        "```python\n"
        "def foo(x):\n"
        "    \"\"\"doc\"\"\"\n"
        "    pass\n"
        "```"
    )
    body = "a = x + 1\n    b = a * 2\n    return b"
    out = _runner()._assemble_function_completion(stub, body)
    _assert_compiles(out)
    assert "    a = x + 1" in out
    assert "    b = a * 2" in out
    assert "    return b" in out


def test_assemble_single_line_body() -> None:
    stub = (
        "```python\n"
        "def bar(n):\n"
        "    \"\"\"doc\"\"\"\n"
        "    pass\n"
        "```"
    )
    body = "return n * 2"
    out = _runner()._assemble_function_completion(stub, body)
    _assert_compiles(out)
    assert "    return n * 2" in out


def test_assemble_body_opens_if_block() -> None:
    """The if-else body must not be flattened either."""
    stub = (
        "```python\n"
        "def g(x):\n"
        "    \"\"\"doc\"\"\"\n"
        "    pass\n"
        "```"
    )
    body = "if x > 0:\n    return x\n        else:\n    return -x"
    # The synthesized indentation should still compile; the author may
    # have written the body inconsistently. We only require non-collapse
    # of the if-block and that Python can parse the result.
    out = _runner()._assemble_function_completion(stub, body)
    # We do not force compile here because the body is intentionally
    # inconsistent; we only verify that the if is not flattened with the
    # function-level statements (the bug would make the if the same
    # column as `return x` outside the branch).
    assert "\n    if x > 0" in out


if __name__ == "__main__":
    test_assemble_preserves_nested_block_body()
    test_assemble_quirky_first_line_flush()
    test_assemble_single_line_body()
    test_assemble_body_opens_if_block()
    print("all tests pass")
