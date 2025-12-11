import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synthetic_data.utils import FunctionExtractor


def test_extract_simple_function():
    """Test extracting a simple function."""
    code = """
from qiskit import QuantumCircuit

def create_bell_state():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc
"""

    functions = FunctionExtractor.extract_functions(code)

    assert len(functions) == 1
    assert functions[0].name == "create_bell_state"
    assert "from qiskit import QuantumCircuit" in functions[0].imports
    assert "def create_bell_state():" in functions[0].signature
    print("✓ Simple function extraction works")


def test_extract_function_with_arguments():
    """Test extracting function with arguments."""
    code = """
from qiskit import QuantumCircuit

def random_number_generator_unsigned_8bit(n):
    circuit = QuantumCircuit(8)
    circuit.h(range(8))
    circuit.measure_all()
    return circuit
"""

    functions = FunctionExtractor.extract_functions(code)

    assert len(functions) == 1
    assert functions[0].name == "random_number_generator_unsigned_8bit"
    assert "n" in functions[0].signature
    print("✓ Function with arguments extraction works")


def test_extract_multiple_functions():
    """Test extracting multiple functions."""
    code = """
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def create_circuit(n):
    return QuantumCircuit(n)

def get_statevector(circuit):
    return Statevector.from_instruction(circuit)
"""

    functions = FunctionExtractor.extract_functions(code)

    assert len(functions) == 2
    function_names = [f.name for f in functions]
    assert "create_circuit" in function_names
    assert "get_statevector" in function_names
    print("✓ Multiple function extraction works")


def test_create_function_stub():
    """Test creating function stub."""
    code = """
from qiskit import QuantumCircuit

def create_bell_state():
    '''Create a Bell state circuit.'''
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc
"""

    func = FunctionExtractor.extract_primary_function(code)
    stub = FunctionExtractor.create_function_stub(func)

    assert "from qiskit import QuantumCircuit" in stub
    assert "def create_bell_state():" in stub
    assert "pass" in stub
    assert "qc.h(0)" not in stub  # Body should be removed
    print("✓ Function stub creation works")


def test_code_to_humaneval_format():
    """Test converting code to HumanEval format."""
    code = """
from qiskit.dagcircuit import DAGCircuit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag

def bell_dag():
    q = QuantumRegister(3, 'q')
    c = ClassicalRegister(3, 'c')
    circ = QuantumCircuit(q, c)
    circ.h(q[0])
    circ.cx(q[0], q[1])
    circ.measure(q[0], c[0])
    dag = circuit_to_dag(circ)
    return dag
"""

    task = "Construct a DAG circuit for a 3-qubit Quantum Circuit with the bell state applied on qubit 0 and 1"

    result = FunctionExtractor.code_to_humaneval_format(code, task)

    assert "prompt" in result
    assert "canonical_solution" in result
    assert "entry_point" in result
    assert "imports" in result

    assert "bell_dag" in result["prompt"]
    assert "with no arguments" in result["prompt"]
    assert result["entry_point"] == "bell_dag"
    assert "def bell_dag():" in result["canonical_solution"]
    print("✓ HumanEval format conversion works")


def test_extract_primary_function():
    """Test extracting primary (longest) function."""
    code = """
from qiskit import QuantumCircuit

def short():
    return QuantumCircuit(1)

def very_long_function_with_many_operations():
    qc = QuantumCircuit(5)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    return qc
"""

    primary = FunctionExtractor.extract_primary_function(code)

    assert primary is not None
    assert primary.name == "very_long_function_with_many_operations"
    print("✓ Primary function extraction works")


def test_function_with_docstring():
    """Test function with docstring."""
    code = '''
from qiskit import QuantumCircuit

def create_ghz(n_qubits):
    """
    Create a GHZ state circuit.
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        QuantumCircuit with GHZ state
    """
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    qc.cx(0, range(1, n_qubits))
    return qc
'''

    functions = FunctionExtractor.extract_functions(code)

    assert len(functions) == 1
    assert functions[0].docstring is not None
    assert "GHZ state" in functions[0].docstring
    assert "Args:" in functions[0].docstring
    print("✓ Function with docstring extraction works")


def test_invalid_code():
    """Test handling invalid code."""
    code = """
def broken syntax here
    print('invalid')
"""

    functions = FunctionExtractor.extract_functions(code)

    assert len(functions) == 0  
    print("✓ Invalid code handling works")


def main():
    """Run all tests."""
    print("Testing FunctionExtractor Utility")

    tests = [
        test_extract_simple_function,
        test_extract_function_with_arguments,
        test_extract_multiple_functions,
        test_create_function_stub,
        test_code_to_humaneval_format,
        test_extract_primary_function,
        test_function_with_docstring,
        test_invalid_code,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} error: {e}")
            failed += 1

    print(f"Results: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
