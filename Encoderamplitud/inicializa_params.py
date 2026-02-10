import numpy as np
import torch


def inic_params(block_size, resize_dim, n_qubits, device):
    params = {}

    for i in range(0, resize_dim[0], block_size):
        for j in range(0, resize_dim[1], block_size):

            params_enc = torch.nn.Parameter(
                torch.rand(n_qubits, device=device) * 2 * np.pi
            )
            params[(i, j)] = (params_enc)

    return params


def crear_optimizador(optimizer_name, params, tasa_de_aprendizaje):
    all_params = []

    for params_enc in params.values():
        all_params.append(params_enc)
        
    if optimizer_name == "Adam":
        return torch.optim.Adam(all_params, lr=tasa_de_aprendizaje)
    elif optimizer_name == "GradientDescent":
        return torch.optim.SGD(all_params, lr=tasa_de_aprendizaje)
    else:
        raise ValueError(f"Optimizador desconocido: {optimizer_name}")