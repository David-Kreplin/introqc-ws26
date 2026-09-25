import numpy as np
from qiskit import QuantumCircuit

from .statevector import sv_array


def print_grover_states(quantum_circuit, num_data_qubits, cutoff=None):
    """
    Prints all basis states of a quantum circuit's statevector with amplitudes above a given threshold.

    Args:
        quantum_circuit (QuantumCircuit): The quantum circuit whose statevector will be analyzed.
        num_data_qubits (int): The number of qubits used for data (excluding ancilla qubits).
        cutoff (float, optional): The minimum amplitude magnitude for a state to be printed.
            Defaults to 1e-10 if not provided.

    Returns:
        None
    """
    num_qubits = quantum_circuit.num_qubits
    statevector = sv_array(quantum_circuit)
    if cutoff is None:
        print("Found basis states with non-zero amplitudes:")
        cutoff = 1e-10
    else:
        print(f"Found basis states with amplitudes greater than {cutoff}:")
    for i, amplitude in enumerate(statevector):
        if np.abs(amplitude) > cutoff:
            bitstr = format(i, f"0{num_qubits}b")
            if num_qubits != num_data_qubits:
                print(
                    f"{bitstr[:num_data_qubits]} {bitstr[num_data_qubits:]}: {np.real(amplitude):5.2f}  Probability: {np.square(np.real(amplitude)):5.2f}"
                )
            else:
                print(
                    f"{bitstr}: {np.real(amplitude):5.2f}  Probability: {np.square(np.real(amplitude)):5.2f}"
                )


def initial_state(num_data_qubits, num_ancilla_qubits):
    """
    Creates the initial quantum state for Grover’s algorithm by applying Hadamard gates
    to all data qubits to generate a uniform superposition.

    Args:
        num_data_qubits (int): Number of data qubits to be placed in superposition.
        num_ancilla_qubits (int): Number of ancilla qubits to be appended to the circuit.

    Returns:
        QuantumCircuit: A quantum circuit representing the initialized state.
    """
    list_of_qubits = list(range(num_data_qubits))
    qc_initial_state = QuantumCircuit(num_data_qubits + num_ancilla_qubits)
    qc_initial_state.h(list_of_qubits)
    return qc_initial_state


def amplitude_amplification(num_data_qubits, num_ancilla_qubits):
    """
    Constructs the amplitude amplification circuit used in Grover’s algorithm.
    This circuit applies a sequence of Hadamard, X, and multi-controlled-X gates
    to amplify the probability of marked states.

    Args:
        num_data_qubits (int): Number of qubits representing the search space.
        num_ancilla_qubits (int): Number of ancillary qubits used for computation.

    Returns:
        QuantumCircuit: The amplitude amplification circuit.
    """
    qc_amplification = QuantumCircuit(num_data_qubits + num_ancilla_qubits)
    list_of_qubits = list(range(num_data_qubits))
    qc_amplification.h(list_of_qubits)
    qc_amplification.x(list_of_qubits)
    qc_amplification.h(num_data_qubits - 1)
    qc_amplification.mcx(list_of_qubits[:-1], num_data_qubits - 1)
    qc_amplification.h(num_data_qubits - 1)
    qc_amplification.x(list_of_qubits)
    qc_amplification.h(list_of_qubits)

    qc_amplification.rz(2 * np.pi, 0)

    return qc_amplification


def oracle_sudoku():
    """
    Builds the oracle circuit for a simplified 2x2 Sudoku problem.
    The oracle encodes constraints for Sudoku rules using multi-controlled Toffoli gates,
    and flips the phase of the solution state.

    Qubit layout:
        - Qubits 0–3: Sudoku cell variables
        - Qubits 4–10: Ancilla qubits for intermediate logical checks
        - Qubit 11: Oracle output qubit

    Constraints checked:
        - q0q1 == 01
        - q2q3 == 01
        - q0q1 == 11
        - q2q3 == 11
        - q0 == q2
        - q1 == q3
        - All conditions must be true to flip the oracle output

    Args:
        None

    Returns:
        QuantumCircuit: The constructed oracle circuit for the Sudoku problem.
    """
    qc_compute = QuantumCircuit(12)

    # Ancilla 1 (Qubit 4) checks q0q1 == 01
    qc_compute.x(0)
    qc_compute.ccx(0, 1, 4)
    qc_compute.x(0)

    # Ancilla 2 (Qubit 5) checks q2q3 == 01
    qc_compute.x(2)
    qc_compute.ccx(2, 3, 5)
    qc_compute.x(2)

    # Ancilla 3 (Qubit 6) checks q0q1 == 11
    qc_compute.ccx(0, 1, 6)

    # Ancilla 4 (Qubit 7) checks q2q3 == 11
    qc_compute.ccx(2, 3, 7)

    # Ancilla 5 (Qubit 8) checks q0 == q2
    qc_compute.cx(0, 8)
    qc_compute.cx(2, 8)
    qc_compute.x(8)

    # Ancilla 6 (Qubit 9) checks q1 == q3
    qc_compute.cx(1, 9)
    qc_compute.cx(3, 9)
    qc_compute.x(9)

    # Check whether both equality checks (Ancilla 5 and 6) are satisfied
    qc_compute.ccx(8, 9, 10)

    # Invert all query results -> correct solution when all are 1
    qc_compute.x([4, 5, 6, 7, 10])

    # Multi-controlled CNOT with the oracle qubit as the target
    qc_compute.mcx([4, 5, 6, 7, 10], 11)

    qc_oracle = QuantumCircuit(12)
    qc_oracle.compose(qc_compute, inplace=True)
    qc_oracle.z(11)
    qc_oracle.compose(qc_compute.inverse(), inplace=True)

    return qc_oracle


def grover_ones(num_data_qubits: int, num_steps: int) -> QuantumCircuit:
    """Implementation of the Grover algorithm that finds the |1...11> state

    Args:
        num_data_qubits (int): Number of data qubits
        num_steps (int): Number of Grover-steps

    Returns:
        Quantum circuit of the oracle
    """

    # Create a quantum circuit with the given number of data qubits
    quantum_circuit = QuantumCircuit(num_data_qubits)

    # Prepare the initial uniform superposition (usually |s⟩ state)
    quantum_circuit.compose(initial_state(num_data_qubits, 0), inplace=True)

    # Define the control qubits (all except the last qubit, which acts as target)
    control_qubits = list(range(num_data_qubits - 1))

    # Repeat the Grover iteration (oracle + amplitude amplification)
    for i in range(num_steps):

        # --- Oracle step ---
        # The oracle marks the "solution" state by flipping its phase.
        # Here represented by a controlled multi-qubit X (inversion) gate.
        quantum_circuit.h(num_data_qubits - 1)
        quantum_circuit.mcx(control_qubits, num_data_qubits - 1)
        quantum_circuit.h(num_data_qubits - 1)

        # --- Diffusion (Amplitude amplification) step ---
        # Reflects the state vector about the average amplitude.
        quantum_circuit.compose(
            amplitude_amplification(num_data_qubits, 0), inplace=True
        )

    # Return the constructed Grover circuit
    return quantum_circuit


def grover_ones_ancilla(num_data_qubits, num_steps):
    """Implementation of the Grover algorithm that finds the |1...11> state using an Ancilla-Qubit

    Args:
        num_data_qubits (int): Number of data qubits
        num_steps (int): Number of Grover-steps

    Returns:
        Quantum circuit of the oracle
    """

    # Create a quantum circuit with one extra ancilla qubit
    quantum_circuit = QuantumCircuit(num_data_qubits + 1)

    # Prepare the initial state, e.g., uniform superposition of all data qubits
    # and the ancilla initialized to |1⟩ (controlled oracle requires ancilla)
    quantum_circuit.compose(initial_state(num_data_qubits, 1), inplace=True)

    # Define the control qubits (all data qubits)
    control_qubits = list(range(num_data_qubits))

    # Perform the specified number of Grover iterations
    for i in range(num_steps):

        # --- Oracle step ---
        # Flip the phase of the |11...1> state using the ancilla qubit.
        # The sequence mcx -> z -> mcx implements a conditional phase inversion.
        quantum_circuit.mcx(control_qubits, num_data_qubits)  # Controlled-X on ancilla
        quantum_circuit.z(num_data_qubits)  # Phase flip on ancilla
        quantum_circuit.mcx(control_qubits, num_data_qubits)  # Undo the controlled-X

        # --- Diffusion (Amplitude amplification) step ---
        # Reflects the state about the mean amplitude to amplify the marked state.
        quantum_circuit.compose(
            amplitude_amplification(num_data_qubits, 1), inplace=True
        )

    # Return the constructed Grover circuit with ancilla
    return quantum_circuit
