from typing import Union
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2
from qiskit_aer import Aer
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit.primitives import BackendEstimatorV2, StatevectorEstimator
from qiskit.compiler import transpile

from .statevector import sv_dict
from .sampling import (
    sample_from_circuit_backend,
    sample_from_circuit,
    measurements_to_probabilities,
)


def qiskit_expectation_value_hardware(
    quantum_circuit: QuantumCircuit,
    observable: SparsePauliOp,
    num_shots: Union[None, int] = None,
) -> dict:
    """
    Executes a quantum circuit with a specified number of measurements (shots)
    and returns the measurement results as a dictionary.

    Args:
        quantum_circuit (QuantumCircuit): The quantum circuit to be executed.
        num_shots (int): The number of measurements (shots) to be performed.

    Returns:
        dict: A dictionary with the measurement results.
    """

    # Initialize the sampler with 100 measurements (shots)
    backend = FakeTorino()
    return qiskit_expectation_value_backend(
        quantum_circuit, observable, backend, num_shots
    )


def qiskit_expectation_value_backend(
    quantum_circuit: QuantumCircuit,
    observable: SparsePauliOp,
    backend,
    num_shots: Union[None, int] = None,
) -> dict:
    """
    Executes a quantum circuit with a specified number of measurements (shots)
    and returns the measurement results as a dictionary.

    Args:
        quantum_circuit (QuantumCircuit): The quantum circuit to be executed.
        num_shots (int): The number of measurements (shots) to be performed.

    Returns:
        dict: A dictionary with the measurement results.
    """

    # Initialize the sampler with 100 measurements (shots)
    estimator = EstimatorV2(
        mode=backend, options={"default_precision": 1 / np.sqrt(num_shots)}
    )

    # Execute the quantum circuit with measurements
    result = estimator.run([(quantum_circuit, observable)]).result()

    return float(result[0].data.evs)


def qiskit_expectation_value(
    quantum_circuit: QuantumCircuit,
    observable: Union[SparsePauliOp, str],
    num_shots: Union[None, int] = None,
) -> dict:
    """
    Executes a quantum circuit with a specified number of measurements (shots)
    and returns the measurement results as a dictionary.

    Args:
        quantum_circuit (QuantumCircuit): The quantum circuit to be executed.
        num_shots (int): The number of measurements (shots) to be performed.

    Returns:
        dict: A dictionary with the measurement results.
    """

    if isinstance(observable, str):
        observable = SparsePauliOp(observable[::-1])

    if num_shots == 0 or num_shots is None:
        # Initialize the sampler with 100 measurements (shots)
        estimator = StatevectorEstimator()
    else:
        # Initialize the sampler with 100 measurements (shots)
        backend = Aer.get_backend("aer_simulator")
        estimator = BackendEstimatorV2(
            backend=backend, options={"default_precision": 1 / np.sqrt(num_shots)}
        )

    # Execute the quantum circuit with measurements
    result = estimator.run([(quantum_circuit, observable)]).result()

    return result[0].data.evs


def Z_expectation(
    pauli_string: str,
    quantum_circuit: QuantumCircuit,
    shots: Union[None, int] = None,
    backend=None,
) -> float:
    """
    Computes the expectation value of a multi-qubit Z Pauli-string operator from a quantum circuit.

    Args:
        pauli_string (str): A string consisting of "I" and "Z" characters representing the Pauli operator.
        quantum_circuit (QuantumCircuit): The quantum circuit to be evaluated.
        shots (int or None): The number of shots for measurement. If None, statevector simulation is used.
        backend: The backend to use for execution if shots is specified.

    Returns:
        float: The expectation value of the given Pauli Z string operator.
    """

    if shots is None:
        # No shots specified, use statevector simulation to compute exact probabilities
        probabilities = sv_dict(quantum_circuit)
    else:
        # Shots specified, perform measurements
        measured_quantum_circuit = quantum_circuit.copy()
        measured_quantum_circuit.measure_all()

        if backend is None:
            # No backend specified, use local sampling from the exact simulator
            measurements = sample_from_circuit(measured_quantum_circuit, shots)
        else:
            # Backend specified, use the provided backend for sampling (fake or real hardware)
            measurements = sample_from_circuit_backend(
                measured_quantum_circuit, shots, backend
            )

        # Convert measurement counts to probabilities
        probabilities = measurements_to_probabilities(measurements)

    # Calculate the expectation value
    expectation_value = 0
    for m in probabilities.keys():
        fac = 1.0
        for i, s in enumerate(pauli_string):
            if s == "Z" and m[i] == "1":
                fac = fac * (-1.0)

        expectation_value += probabilities[m] * fac

    return expectation_value


def single_expectation_value(
    pauli_string: str,
    quantum_circuit: QuantumCircuit,
    shots: Union[None, int] = None,
    backend=None,
) -> float:
    """
    Computes the expectation value of a single Pauli string operator build from (X, Y, Z, I) from a quantum circuit.

    Args:
        pauli_string (str): A string consisting of "X", "Y", "Z", and "I" characters representing the Pauli operator.
        quantum_circuit (QuantumCircuit): The quantum circuit to be evaluated.
        shots (int or None): The number of shots for measurement. If None, statevector simulation is used.
        backend: The backend to use for execution if shots is specified.

    Returns:
        float: The expectation value of the given Pauli string operator.
    """

    modified_quantum_circuit = quantum_circuit.copy()

    Z_string = ""
    for i, s in enumerate(pauli_string):
        if s == "X":
            modified_quantum_circuit.h(i)
            Z_string += "Z"
        elif s == "Y":
            modified_quantum_circuit.sdg(i)
            modified_quantum_circuit.h(i)
            Z_string += "Z"
        elif s == "Z":
            Z_string += "Z"
        elif s == "I":
            Z_string += "I"
        else:
            raise ValueError("Unknown PauliGate:", s)

    return Z_expectation(Z_string, modified_quantum_circuit, shots, backend)


def expectation_value(
    observable: SparsePauliOp,
    quantum_circuit: QuantumCircuit,
    shots: Union[None, int] = None,
    backend=None,
) -> float:
    """
    Computes the expectation value of a given SparsePauliOp observable from a quantum circuit.

    Args:
        observable (SparsePauliOp): The observable to be evaluated.
        quantum_circuit (QuantumCircuit): The quantum circuit to be evaluated.
        shots (int or None): The number of shots for measurement. If None, statevector simulation is used.
        backend: The backend to use for execution if shots is specified.

    Returns:
        float: The expectation value of the given observable.
    """

    expectation_value = 0.0

    for o in observable:
        expectation_value += (
            single_expectation_value(str(o.paulis[0]), quantum_circuit, shots, backend)
            * o.coeffs[0]
        )

    return float(np.real_if_close(expectation_value))


def standard_deviation(
    observable: SparsePauliOp, quantum_circuit: QuantumCircuit, shots: int
):
    """
    Computes the standard deviation of a given SparsePauliOp observable from a quantum circuit.

    Args:
        observable (SparsePauliOp): The observable to be evaluated.
        quantum_circuit (QuantumCircuit): The quantum circuit to be evaluated.
        shots (int): The number of shots for computing the standard deviation.

    Returns:
        float: The standard deviation of the given observable.
    """

    observable_squared = observable.power(2).simplify()
    exp_values_squared = expectation_value(
        observable_squared, quantum_circuit, shots=None
    )
    exp_values = expectation_value(observable, quantum_circuit, shots=None)

    std = np.sqrt((exp_values_squared - np.square(exp_values)) / shots)

    return std


def evaluate_expectation_value(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    parameters: np.array = None,
    shots: Union[None, int] = None,
    backend=None,
) -> float:
    """
    Evaluates the expectation value of the given observable with respect to the state prepared by the circuit with the specified parameters.

    Args:
        circuit: The parameterized quantum circuit.
        observable: The observable to measure.
        parameters: The parameters to bind to the circuit.
        shots: Number of shots for execution. If None, a statevector simulator is used.
        backend: The backend to use for execution. If None, perfect backend is used.

    Returns:
        float: The expectation value of the observable.

    """
    if shots is None:
        estimator = StatevectorEstimator()
        circuit_run = circuit
        observable_run = observable
    elif backend is None:
        backend = Aer.get_backend("aer_simulator")
        estimator = EstimatorV2(
            mode=backend, options={"default_precision": 1 / np.sqrt(shots)}
        )
        circuit_run = circuit
        observable_run = observable

    else:

        circuit_run = transpile(circuit, backend=backend)
        observable_run = observable.apply_layout(circuit_run.layout)

        estimator = EstimatorV2(
            mode=backend, options={"default_precision": 1 / np.sqrt(shots)}
        )

    result = estimator.run([(circuit_run, observable_run, parameters)]).result()
    return float(result[0].data.evs)


def gradient_finite_differences(
    quantum_circuit: QuantumCircuit,
    observable: SparsePauliOp,
    parameters: np.ndarray,
    delta_h: float = 0.01,
    shots: Union[int, None] = None,
    backend=None,
) -> np.array:
    """Computes the gradient of the expectation value function using finite differences.

    Args:
        quantum_circuit (QuantumCircuit): The parameterized quantum circuit.
        observable (SparsePauliOp): The observable for which the expectation value is computed.
        parameters (np.ndarray): The current parameters of the quantum circuit.
        delta_h (float): The small perturbation used for finite differences.
        shots (Union[int, None]): Number of shots for measurement. If None, statevector simulation is used.
        backend: The backend to use for execution. If None, perfect backend is used.

    Returns:
        np.array: The gradient vector of the expectation value with respect to the parameters.
    """

    gradient = []

    for p in range(len(parameters)):
        params_plus = parameters.copy()
        params_plus[p] += delta_h
        exp_plus = evaluate_expectation_value(
            quantum_circuit, observable, params_plus, backend=backend, shots=shots
        )

        params_minus = parameters.copy()
        params_minus[p] -= delta_h
        exp_minus = evaluate_expectation_value(
            quantum_circuit, observable, params_minus, backend=backend, shots=shots
        )

        gradient.append((exp_plus - exp_minus) / (2 * delta_h))

    return np.array(gradient)


def gradient_parameter_shift_rule(
    quantum_circuit: QuantumCircuit,
    observable: SparsePauliOp,
    parameters: np.ndarray,
    shots: Union[int, None] = None,
    backend=None,
) -> np.array:
    """Computes the gradient of the expectation value function using the parameter-shift rule.

    Args:
        quantum_circuit (QuantumCircuit): The parameterized quantum circuit.
        observable (SparsePauliOp): The observable for which the expectation value is computed.
        parameters (np.ndarray): The current parameters of the quantum circuit.
        shots (Union[int, None]): Number of shots for measurement. If None, statevector simulation is used.
        backend: The backend to use for execution. If None, perfect backend is used.

    Returns:
        np.array: The gradient vector of the expectation value with respect to the parameters.
    """

    gradient = []

    for p in range(len(parameters)):
        shift = np.pi / 2

        params_plus = parameters.copy()
        params_plus[p] += shift
        exp_plus = evaluate_expectation_value(
            quantum_circuit, observable, params_plus, backend=backend, shots=shots
        )

        params_minus = parameters.copy()
        params_minus[p] -= shift
        exp_minus = evaluate_expectation_value(
            quantum_circuit, observable, params_minus, backend=backend, shots=shots
        )

        gradient.append((exp_plus - exp_minus) / 2)

    return np.array(gradient)
