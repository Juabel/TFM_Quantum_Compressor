import numpy as np
import torch


def inic_params(block_size, resize_dim, n_qubits_dec, n_qubits, device="cpu"):
    params = {}

    for i in range(0, resize_dim[0], block_size):
        for j in range(0, resize_dim[1], block_size):

            params_enc = torch.nn.Parameter(
                torch.rand(n_qubits, device=device) * 2 * np.pi
            )

            params_dec = torch.nn.Parameter(
                torch.rand(n_qubits_dec, device=device) * 2 * np.pi
            )


            params[(i, j)] = (params_enc, params_dec)

    return params


def crear_optimizador(optimizer_name, params, tasa_de_aprendizaje):
    all_params = []

    for params_enc, params_dec in params.values():
        all_params.append(params_enc)
        all_params.append(params_dec)

    if optimizer_name == "Adam":
        return torch.optim.Adam(all_params, lr=tasa_de_aprendizaje)
    elif optimizer_name == "GradientDescent":
        return torch.optim.SGD(all_params, lr=tasa_de_aprendizaje)
    else:
        raise ValueError(f"Optimizador desconocido: {optimizer_name}")
    
def inic_params_angle(block_size, resize_dim, n_qubits, n_layers):
    params = {}
    for i in range(0, resize_dim[0], block_size):
        for j in range(0, resize_dim[1], block_size):
            params_enc = torch.nn.Parameter(
                0.01 * torch.randn(n_layers, n_qubits, 3)
            )
            
            params_dec = torch.nn.Parameter(
                0.01 * torch.randn(n_layers, n_qubits, 3)
            )
            print(params_dec)
            params[(i, j)] = (params_enc, params_dec)

    return params



#EN ESTE CASO, LOS PARAMETROS DEL DECODER NO LOS TOCAMOS, PERO NO HAY PARAMETROS ENCODER, ESTAN LOS VALORES DE THETA Y PHI, QUE SON LOS PARAMETROS QUE VAMOS A ENTRENAR DE LA TECNICA DE ENCODING
def inic_params_SQ(block_size, resize_dim, n_layers):
    params = {}
    
    n_features = block_size * block_size  # nº de píxeles por bloque

    for i in range(0, resize_dim[0], block_size):
        for j in range(0, resize_dim[1], block_size):

            theta = torch.nn.Parameter(
                0.1 * torch.randn(n_layers, n_features)
            )

            phi = torch.nn.Parameter(
                0.1 * torch.randn(n_layers, n_features)
            )

            # decoder (NO tocar)
            params_dec = torch.nn.Parameter(
                0.01 * torch.randn(n_layers, 2, 3)
            )

            print(params_dec)

            params[(i, j)] = (theta, phi, params_dec)

    return params


################A REVISAR################



def crear_optimizador_SQ(optimizer_name, params, tasa_de_aprendizaje):
    all_params = []

    for theta, phi, params_dec in params.values():
        all_params.append(theta)
        all_params.append(phi)
        all_params.append(params_dec)

    if optimizer_name == "Adam":
        return torch.optim.Adam(all_params, lr=tasa_de_aprendizaje)
    elif optimizer_name == "GradientDescent":
        return torch.optim.SGD(all_params, lr=tasa_de_aprendizaje)
    else:
        raise ValueError(f"Optimizador desconocido: {optimizer_name}")