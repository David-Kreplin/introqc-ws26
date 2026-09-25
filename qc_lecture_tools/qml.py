import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from squlearn import Executor
from squlearn.encoding_circuit import QiskitEncodingCircuit
from squlearn.observables import CustomObservable
from squlearn.qnn.lowlevel_qnn import LowLevelQNN


def evaluate_qnn(
    qiskit_circuit: QuantumCircuit,
    observable: SparsePauliOp,
    x_values,
) -> np.ndarray:
    """
    Evaluate the QNN output f(x, p) for one or many x-values.

    Parameters
    ----------
    qiskit_circuit : QuantumCircuit
        Qiskit circuit containing parameters labeled with x and p.
    observable : SparsePauliOp
        Observable to measure.
    x_values : float, list, or np.ndarray
        Input feature value(s). For a 1D problem, this can be a scalar or an array.

    Returns
    -------
    np.ndarray
        Evaluated QNN output.
    """
    encoding_circuit = QiskitEncodingCircuit(
        qiskit_circuit,
        mode="auto",
        feature_label="x",
        parameter_label="p",
    )

    custom_observable = CustomObservable(
        num_qubits=observable.num_qubits,
        operator_string=list(observable.paulis.to_labels()),
        parameterized=False,
    )

    lowlevel_qnn = LowLevelQNN(
        encoding_circuit,
        custom_observable,
        executor=Executor("qulacs"),
        num_features=1,
    )

    p_values = (np.random.rand(encoding_circuit.num_parameters) - 0.5) * 2 * np.pi

    result = lowlevel_qnn.evaluate(
        x_values,
        p_values,
        np.array([]),
        "f",
    )
    return np.asarray(result["f"])