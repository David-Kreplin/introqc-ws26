"""Provided helpers for the Shor-algorithm lab."""

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import QuantumCircuit
from .statevector import sv_dict


def print_basis_state_mapping(
    circuit: QuantumCircuit,
    input_states: list[int],
) -> list[tuple[int, int]]:
    """Print the basis-state mapping produced by a candidate circuit.

    The same circuit is applied independently to every input state.
    """
    if circuit.num_qubits != 4:
        raise ValueError(
            "The lab helper expects exactly 4 qubits for integer states."
        )

    width = circuit.num_qubits
    mappings = []

    for input_integer in input_states:
        # Prepare the input basis state |input_integer>
        input_bitstring = f"{input_integer:0{width}b}"
        test_circuit = QuantumCircuit(width)

        for qubit, bit in enumerate(reversed(input_bitstring)):
            if bit == "1":
                test_circuit.x(qubit)

        # Apply the candidate circuit
        test_circuit.compose(circuit, inplace=True)

        # Use the course helper to read out the resulting basis state
        probabilities = sv_dict(test_circuit.reverse_bits())

        if len(probabilities) != 1:
            raise ValueError(
                "The circuit does not map the input basis state "
                "deterministically to a single output basis state."
            )

        output_bitstring = next(iter(probabilities))
        output_integer = int(output_bitstring, 2)

        mappings.append((input_integer, output_integer))

        print(f"|{input_bitstring}> -> |{output_bitstring}>")

    return mappings


def _inverse_qft(num_qubits: int):
    """Return the inverse QFT gate used by the provided QPE black box."""
    circuit = QuantumCircuit(num_qubits, name="QFT†")

    for qubit in range(num_qubits // 2):
        circuit.swap(qubit, num_qubits - qubit - 1)

    for target in range(num_qubits):
        for control in range(target):
            angle = -np.pi / (2 ** (target - control))
            circuit.cp(angle, control, target)
        circuit.h(target)

    return circuit.to_gate(label="QFT†")


def create_qpe_circuit(
    controlled_modular_function,
    num_ancilla_qubits: int,
    num_target_qubits: int = 4,
) -> QuantumCircuit:
    """Build QPE from a student-supplied controlled modular-power function.

    controlled_modular_function(power) must return a gate, instruction, or
    circuit acting on one control qubit followed by num_target_qubits target
    qubits. The function is called with powers 1, 2, 4, ... .
    """
    if num_ancilla_qubits < 1:
        raise ValueError("num_ancilla_qubits must be at least 1")

    counting = QuantumRegister(num_ancilla_qubits, "ancilla")
    target = QuantumRegister(num_target_qubits, "target")
    measured_phase = ClassicalRegister(num_ancilla_qubits, "c")
    circuit = QuantumCircuit(counting, target, measured_phase)

    circuit.h(counting)
    circuit.x(target[0])

    expected_block_qubits = 1 + num_target_qubits
    for ancilla_index in range(num_ancilla_qubits):
        power = 2**ancilla_index
        controlled_block = controlled_modular_function(power)

        if controlled_block.num_qubits != expected_block_qubits:
            raise ValueError(
                "The controlled modular function must return an operation on "
                f"{expected_block_qubits} qubits: one control followed by "
                f"{num_target_qubits} target qubits."
            )

        if isinstance(controlled_block, QuantumCircuit):
            controlled_block = controlled_block.to_gate(
                label=controlled_block.name
            )

        circuit.append(
            controlled_block,
            [counting[ancilla_index], *target],
        )

    circuit.append(_inverse_qft(num_ancilla_qubits), counting)
    for i in range(num_ancilla_qubits):
        circuit.measure(num_ancilla_qubits-1-i, i)
    return circuit
