import pennylane as qml
from pennylane import numpy as np


def inic_params(block_size, resize_dim, n_qubits_dec, n_qubits):
    params = {}

    for i in range(0, resize_dim[0], block_size):
        for j in range(0, resize_dim[1], block_size):

            # Encoder: un solo vector de parámetros
            params_enc = qml.numpy.array(
                np.random.uniform(0, 2 * np.pi, size=n_qubits),
                requires_grad=True
            )

            # Decoder: un solo vector de parámetros
            params_dec = qml.numpy.array(
                np.random.uniform(0, 2 * np.pi, size=n_qubits_dec),
                requires_grad=True
            )

            params[(i, j)] = (params_enc, params_dec)

    return params
