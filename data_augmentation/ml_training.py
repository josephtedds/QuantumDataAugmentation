from enum import Enum
from math import ceil, gcd
from typing import Dict

SIM_ANSATZE_PER_QUBITS = {
    1: lambda n: {
        "num_rot_gates": 2 * n,
        "num_c_rot_gates": 0,
    },
    2: lambda n: {
        "num_rot_gates": 2 * n,
        "num_c_rot_gates": 0,
    },
    3: lambda n: {
        "num_rot_gates": 2 * n,
        "num_c_rot_gates": n - 1,
    },
    4: lambda n: {
        "num_rot_gates": 2 * n,
        "num_c_rot_gates": n - 1,
    },
    5: lambda n: {
        "num_rot_gates": 4 * n,
        "num_c_rot_gates": n**2 - n,
    },
    6: lambda n: {
        "num_rot_gates": 4 * n,
        "num_c_rot_gates": n**2 - n,
    },
    7: lambda n: {
        "num_rot_gates": 4 * n,
        "num_c_rot_gates": n - 1,
    },
    8: lambda n: {
        "num_rot_gates": 4 * n,
        "num_c_rot_gates": n - 1,
    },
    9: lambda n: {
        "num_rot_gates": n,
        "num_c_rot_gates": 0,
    },
    10: lambda n: {
        "num_rot_gates": 2 * n,
        "num_c_rot_gates": 0,
    },
    11: lambda n: {
        "num_rot_gates": 4 * n - 4,
        "num_c_rot_gates": 0,
    },
    12: lambda n: {
        "num_rot_gates": 4 * n - 4,
        "num_c_rot_gates": 0,
    },
    13: lambda n: {
        "num_rot_gates": 2 * n,
        "num_c_rot_gates": int(n + n / gcd(n, 3)),
    },
    14: lambda n: {
        "num_rot_gates": 2 * n,
        "num_c_rot_gates": int(n + n / gcd(n, 3)),
    },
    15: lambda n: {
        "num_rot_gates": 2 * n,
        "num_c_rot_gates": 0,
    },
    16: lambda n: {
        "num_rot_gates": 2 * n,
        "num_c_rot_gates": n - 1,
    },
    17: lambda n: {
        "num_rot_gates": 2 * n,
        "num_c_rot_gates": n - 1,
    },
    18: lambda n: {
        "num_rot_gates": 2 * n,
        "num_c_rot_gates": n,
    },
    19: lambda n: {
        "num_rot_gates": 2 * n,
        "num_c_rot_gates": n,
    },
}


def sim_ansatz(
    ansatz_idx: int, n_qubits: int = 4, layers: int = 1
) -> Dict[str, int]:
    """Generate a Sim circuit.

    Ansatze are taken from: arXiv:1905.10876

    Parameters
    ----------
    ansatz_idx : int
        Index of the ansatz to use.
    n_qubits : int, optional
        Number of qubits in the circuit, by default 4
    layers : int, optional
        Number of ansatz layers used, by default 1

    Returns
    -------
    Dict[str, int]
        Resource estimator values for Sim circuits with a given number
        of qubits and ansatz repetitions.

    Raises
    ------
    ValueError
        If the number of qubits provided is 1, but an ansatz is
        picked that uses 2-qubit gates.
    """
    if n_qubits == 1 and ansatz_idx != 1:
        raise ValueError(
            "This circuit uses 2 parameter gates, but only 1 qubit was "
            "provided"
        )

    if ansatz_idx == 10:
        base_circuit = {
            "num_rot_gates": n_qubits * (1 + layers),
            "num_c_rot_gates": 0,
        }
        return base_circuit

    base_circuit = SIM_ANSATZE_PER_QUBITS[ansatz_idx](n_qubits)

    for i_key, i_val in base_circuit.items():
        base_circuit[i_key] = layers * i_val
    return base_circuit


class EmbeddingAnsatz(Enum):
    """Embedding method for classical data."""

    ANGLE_EMBEDDING = 0


class DifferentiationMethod(Enum):
    """Differentiation methods for quantum gradients."""

    FINITE_DIFFERENCE = 0
    PARAMETER_SHIFT = 1
    SPSA = 2


class ClassicalDataEmbedding:
    """Classical data embedding resource calculation.

    Attributes
    ----------
    num_qubits: int
        The number of qubits the circuit acts on.
    embedding_ansatz: EmbeddingAnsatz
        Ansatz used to embed classical data.
    """

    def __init__(
        self,
        num_qubits: int,
        embedding_ansatz: EmbeddingAnsatz.ANGLE_EMBEDDING,
    ) -> None:
        self._num_qubits = num_qubits
        self.embedding_ansatz = embedding_ansatz

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits used by the circuit."""
        return self._num_qubits

    @property
    def embedding_ansatz(self) -> EmbeddingAnsatz:
        """Ansatz used to embed classical data."""
        return self._embedding_ansatz

    @embedding_ansatz.setter
    def embedding_ansatz(self, embedding_ansatz: EmbeddingAnsatz):
        if not isinstance(embedding_ansatz, EmbeddingAnsatz):
            raise ValueError("The provided value is not an EmbeddingAnsatz")
        self._embedding_ansatz = embedding_ansatz

    def get_num_gradient_evals_per_datapoint(
        self, diff_method: DifferentiationMethod
    ) -> int:
        """Get the number of gradient evaluations per datapoint.

        This is ONLY implmented for the simple Angle Embedding ansatz
        i.e. a single layer of single parameter rotations.

        Parameters
        ----------
        diff_method : DifferentiationMethod
            Differentation method for gradient calculations.

        Returns
        -------
        int
            Number of additional gradient evaluations per datapoint.

        Raises
        ------
        NotImplementedError
            If the differentiation method given does not exist or is
            unimplemented.
        """
        if diff_method is DifferentiationMethod.SPSA:
            return 2 * self.num_qubits
        elif diff_method is DifferentiationMethod.FINITE_DIFFERENCE:
            return self.num_qubits
        elif diff_method is DifferentiationMethod.PARAMETER_SHIFT:
            return 2 * self.num_qubits
        else:
            raise NotImplementedError(
                "The number of additional quantum circuit executions for this "
                "method is unknown."
            )


class MlTrainingResourceEstimator:
    """Machine learning resource estimator.

    Given a simple circuit with differentiation method one of finite
    difference, parameter shift or SPSA, this provides estimates for the
    number of circuits needed for different size training sets and
    number of epochs completed

    While this is targeted at circuits with only parameterised rotations
    e.g. RX, RY and controlled parameterised rotations e.g. CRX the
    classification into num_rot_gates and num_c_rot_gates only makes
    a difference for parameter shift calculations where the controlled
    gates need 4 extra executions and the standard rotations need 2.

    Attributes
    ----------
    num_rot_gates: int
        The number of rotational gates - these should be X, Y or Z NOT a
        general rotation (which by Euler decomposition is equivalent to
        3 rotational gates in this framework).
    num_c_rot_gates: int
        The number of controlled rotational gates - these should be X, Y
        or Z NOT a general rotation (which by Euler decomposition is
        equivalent to 3 rotational gates in this framework).
    diff_method: DifferentiationMethod
        Differentation method for gradient calculations, one of finite
        difference, parameter shift or SPSA.
    classical_embedding: ClassicalDataEmbedding
        Method used to embed classical data. Currently just a simple
        angle embedding layer.
    """

    def __init__(
        self,
        num_rot_gates: int = 0,
        num_c_rot_gates: int = 0,
        diff_method: DifferentiationMethod = DifferentiationMethod.PARAMETER_SHIFT,
        classical_embedding: ClassicalDataEmbedding = None,
    ):
        self._num_rot_gates = num_rot_gates
        self._num_c_rot_gates = num_c_rot_gates
        self.diff_method = diff_method
        self.classical_embedding = classical_embedding

    @property
    def num_rot_gates(self) -> int:
        """Get the number of RX, RY, RZ gates in the circuit."""
        return self._num_rot_gates

    @property
    def num_c_rot_gates(self) -> int:
        """Get the number of CRX, CRY, CRZ gates in the circuit."""
        return self._num_c_rot_gates

    @property
    def num_parameters(self) -> int:
        """Get the total number of parameters in the circuit."""
        return self.num_rot_gates + self.num_c_rot_gates

    @property
    def diff_method(self) -> DifferentiationMethod:
        """Differentation method for gradient calculations."""
        return self._diff_method

    @diff_method.setter
    def diff_method(self, diff_method: DifferentiationMethod):
        if not isinstance(diff_method, DifferentiationMethod):
            raise ValueError(
                "The provided value is not a DifferentationMethod"
            )
        self._diff_method = diff_method

    @property
    def classical_embedding(self) -> ClassicalDataEmbedding:
        """Classical data embedding method."""
        return self._classical_embedding

    @classical_embedding.setter
    def classical_embedding(self, classical_embedding: ClassicalDataEmbedding):
        if (
            not isinstance(classical_embedding, ClassicalDataEmbedding)
            and classical_embedding is not None
        ):
            raise ValueError(
                "The provided value is not a DifferentationMethod"
            )
        self._classical_embedding = classical_embedding

    def get_num_gradient_evals_per_datapoint(
        self, trainable_pre_quantum_layers: bool = False
    ) -> int:
        """Get the number of gradient evaluations per datapoint.

        This differs for each differentiation method.

        Parameters
        ----------
        trainable_pre_quantum_layers: bool, optional
            Whether there are trainable layers before this quantum layer
            e.g. a classical neural network, by default False

        Returns
        -------
        int
            The number of additional quantum circuit executions to
            evaluate the gradient.
        """
        embedding_gradient_evals = 0

        if trainable_pre_quantum_layers:
            if self.classical_embedding is None:
                raise ValueError("No classical embedding method provided")
            embedding_gradient_evals += (
                self.classical_embedding.get_num_gradient_evals_per_datapoint(
                    self.diff_method
                )
            )

        if self.diff_method is DifferentiationMethod.SPSA:
            return embedding_gradient_evals + 2
        elif self.diff_method is DifferentiationMethod.FINITE_DIFFERENCE:
            return embedding_gradient_evals + self.num_parameters
        elif self.diff_method is DifferentiationMethod.PARAMETER_SHIFT:
            return (
                embedding_gradient_evals
                + 2 * self.num_rot_gates
                + 4 * self.num_c_rot_gates
            )
        else:
            raise NotImplementedError(
                "The number of additional quantum circuit executions for this "
                "method is unknown."
            )

    def estimate_num_training_circuits(
        self, training_set_size: int, n_epochs: int
    ):
        """Estimate the number of circuits needed to train this circuit.

        Parameters
        ----------
        training_set_size : int, optional
            Number of data point in the training set, by default 1000.
        n_epochs : int, optional
            Number of epochs the model is run for, by default 100.

        Returns
        -------
        int
            The total number of circuits run for the training cycle.
        """
        return (
            training_set_size
            * n_epochs
            * (1 + self.get_num_gradient_evals_per_datapoint())
        )

    def estimate_training_time_in_seconds(
        self,
        time_per_circuit_in_seconds: float,
        training_set_size: int = 1000,
        n_epochs: int = 100,
        trainable_pre_quantum_layer: bool = False,
        multiprogramming_factor: int = 1,
    ) -> float:
        """Estimate the time needed to train this circuit.

        Parameters
        ----------
        time_per_circuit_in_seconds : float
            The average time to run a single parameterisation of this
            circuit for a chosen number of shots.
        training_set_size : int, optional
            Number of data point in the training set, by default 1000.
        n_epochs : int, optional
            Number of epochs the model is run for, by default 100.
        trainable_pre_quantum_layers: bool, optional
            Whether there are trainable layers before this quantum layer
            e.g. a classical neural network, by default False.
        multiprogramming_factor: int, optional
            Whether to use multiprogramming techniques to stack multiple
            copies of a circuit onto a large QPU.

        Returns
        -------
        float
            The total time to train the circuit in seconds.
        """
        multi_programming_dataset_size = ceil(
            training_set_size / multiprogramming_factor
        )
        return (
            time_per_circuit_in_seconds
            * multi_programming_dataset_size
            * n_epochs
            * (
                1
                + self.get_num_gradient_evals_per_datapoint(
                    trainable_pre_quantum_layer
                )
            )
        )


if __name__ == "__main__":

    CIRCUIT_EXAMPLES = [
        {
            "sim_circuit": {
                "ansatz_idx": 2,
                "n_qubits": 4,
                "layers": 1,
            },
            "diff_method": DifferentiationMethod.FINITE_DIFFERENCE,
        },
        {
            "sim_circuit": {
                "ansatz_idx": 2,
                "n_qubits": 4,
                "layers": 1,
            },
            "diff_method": DifferentiationMethod.PARAMETER_SHIFT,
        },
        {
            "sim_circuit": {
                "ansatz_idx": 18,
                "n_qubits": 4,
                "layers": 1,
            },
            "diff_method": DifferentiationMethod.FINITE_DIFFERENCE,
        },
        {
            "sim_circuit": {
                "ansatz_idx": 18,
                "n_qubits": 4,
                "layers": 1,
            },
            "diff_method": DifferentiationMethod.PARAMETER_SHIFT,
        },
        {
            "sim_circuit": {
                "ansatz_idx": 7,
                "n_qubits": 10,
                "layers": 4,
            },
            "diff_method": DifferentiationMethod.PARAMETER_SHIFT,
        },
    ]

    for i_circuit_example in CIRCUIT_EXAMPLES:
        # Construct Sim circuit
        i_circ_ansatz_stats = i_circuit_example["sim_circuit"]
        i_circ = sim_ansatz(**i_circ_ansatz_stats)

        # Generate estimator object
        i_resource_estimator = MlTrainingResourceEstimator(
            **i_circ, diff_method=i_circuit_example["diff_method"]
        )
        i_circ_estimate = i_resource_estimator.estimate_num_training_circuits(
            1000, 100
        )
        print(
            f"For a Sim{i_circ_ansatz_stats['ansatz_idx']} ansatz with "
            f"{i_circ_ansatz_stats['n_qubits']} and "
            f"{i_circ_ansatz_stats['layers']} layer(s), if you trained this"
            " for 100 epochs on a training set size of 100, you'd need "
            f"{i_circ_estimate:,}"
            " circuits."
        )
