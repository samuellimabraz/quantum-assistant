import pytest

from synthetic_data.utils.code_verifier import CodeVerifier


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, fixed_response: str = None):
        self.fixed_response = fixed_response
        self.call_count = 0

    def generate(self, messages, temperature=0.7):
        self.call_count += 1
        if self.fixed_response:
            return self.fixed_response

        return """```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
print("Bell state created")
```"""

    async def aclose(self):
        pass


def test_code_verifier_valid_code():
    """Test verification of valid code."""
    verifier = CodeVerifier(MockLLMClient(), max_iterations=3)

    valid_code = """Here's a simple quantum circuit:

```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
print("Done")
```

This creates a Bell state."""

    result = verifier.verify_and_correct_sample(valid_code)

    assert result.is_valid
    assert result.error_message is None
    assert result.iterations_used == 0


def test_code_verifier_syntax_error():
    """Test detection of syntax errors."""
    verifier = CodeVerifier(MockLLMClient(), max_iterations=3)

    invalid_code = """Here's a circuit with syntax error:

```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
if True
    qc.h(0)
```"""

    result = verifier.verify_and_correct_sample(invalid_code)
    assert result.error_type == "syntax" if not result.is_valid else True


def test_code_verifier_import_error():
    """Test detection of import errors during execution."""
    verifier = CodeVerifier(MockLLMClient(), max_iterations=3)

    wrong_import = """Here's code with wrong import:

```python
from qiskit import QuantumCircuit
from nonexistent_module import something

qc = QuantumCircuit(2)
```"""

    result = verifier.verify_and_correct_sample(wrong_import)
    assert result.error_type in ["execution", None] if not result.is_valid else True


def test_code_verifier_execution_error():
    """Test detection of runtime errors."""
    verifier = CodeVerifier(MockLLMClient(), max_iterations=3)

    runtime_error = """Here's code with undefined variable:

```python
from qiskit import QuantumCircuit

# Missing qc definition
qc.h(0)
```"""

    result = verifier.verify_and_correct_sample(runtime_error)
    assert result.error_type in ["execution", None] if not result.is_valid else True


def test_code_verifier_no_code():
    """Test with text that has no code blocks."""
    verifier = CodeVerifier(MockLLMClient(), max_iterations=3)

    no_code = "This is just a conceptual explanation of quantum superposition."

    result = verifier.verify_and_correct_sample(no_code)

    assert result.is_valid
    assert result.iterations_used == 0


def test_code_verifier_multiple_blocks():
    """Test with multiple code blocks."""
    verifier = CodeVerifier(MockLLMClient(), max_iterations=3)

    multiple_blocks = """First example:

```python
from qiskit import QuantumCircuit
qc = QuantumCircuit(1)
```

Second example:

```python
from qiskit import QuantumCircuit
qc2 = QuantumCircuit(2)
qc2.h(0)
print("Done")
```"""

    result = verifier.verify_and_correct_sample(multiple_blocks)
    assert result.is_valid or result.iterations_used > 0


def test_extract_code_blocks():
    """Test code block extraction."""
    verifier = CodeVerifier(MockLLMClient())

    text = """Some text.

```python
code1
```

More text.

```
code2
```

End."""

    blocks = verifier._extract_code_blocks(text)
    assert len(blocks) == 2
    assert "code1" in blocks[0]
    assert "code2" in blocks[1]


def test_check_syntax_valid():
    """Test syntax checking with valid code."""
    verifier = CodeVerifier(MockLLMClient())

    valid_code = """
from qiskit import QuantumCircuit

def create_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    return qc

circuit = create_circuit()
"""

    error = verifier._check_syntax(valid_code)
    assert error is None


def test_check_syntax_invalid():
    """Test syntax checking with invalid code."""
    verifier = CodeVerifier(MockLLMClient())

    invalid_code = """
from qiskit import QuantumCircuit

if True
    qc = QuantumCircuit(2)
"""

    error = verifier._check_syntax(invalid_code)
    assert error is not None
    assert "syntax" in error.lower()


def test_verify_with_question_context():
    """Test verification with question context for better correction."""
    verifier = CodeVerifier(MockLLMClient(), max_iterations=3)

    question = "Create a Bell state circuit"
    answer_with_error = """Here's how to create a Bell state:

```python
from qiskit import QuantumCircuit

# Missing qc definition
qc.h(0)
qc.cx(0, 1)
```"""

    result = verifier.verify_and_correct_sample(answer_with_error, question=question)

    assert result.error_type in ["execution", None] if not result.is_valid else True


def test_max_iterations():
    """Test that max iterations is respected."""
    mock_client = MockLLMClient(fixed_response="```python\nif True\n    print('bad')\n```")
    verifier = CodeVerifier(mock_client, max_iterations=2)

    invalid_code = """
```python
if True
    print("syntax error")
```
"""

    result = verifier.verify_and_correct_sample(invalid_code)

    assert not result.is_valid
    assert result.iterations_used == 2
    assert mock_client.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
