import pennylane as qml
from pennylane import numpy as np


def inic_params(params_dec, block_size, resize_dim, n_qubits_dec, n_qubits, num_layers):

    params_enc = {}
    params_dec = {}

    for i in range(0, resize_dim[0], block_size):
        for j in range(0, resize_dim[1], block_size):
            # Encoder: lista de arrays por capa
            params_enc_por_bloque = [
                qml.numpy.array(np.random.uniform(0, 2*np.pi, size=n_qubits), requires_grad=True)
                for _ in range(num_layers)
            ]
            params_enc[(i,j)] = params_enc_por_bloque

            # Decoder: lista de arrays por capa
            params_dec_por_bloque = [
                qml.numpy.array(np.random.uniform(0, 2*np.pi, size=n_qubits_dec), requires_grad=True)
                for _ in range(num_layers)
            ]
            params_dec[(i,j)] = params_dec_por_bloque

    return params_dec, params_enc, 