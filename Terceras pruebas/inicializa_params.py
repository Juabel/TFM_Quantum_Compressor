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


def crear_optimizador(optimizer_name, params_enc, tasa_de_aprendizaje):

    if optimizer_name == "Adam":
        return torch.optim.Adam([params_enc], lr=tasa_de_aprendizaje)
    elif optimizer_name == "GradientDescent":
        return torch.optim.SGD([params_enc], lr=tasa_de_aprendizaje)
    else:
        raise ValueError(f"Optimizador desconocido: {optimizer_name}")
    
def inic_params_angle(n_qubits, n_layers, ansatz):

    if ansatz == "StronglyEntangling":

        params_enc = torch.nn.Parameter(
            0.01 * torch.randn(n_layers, n_qubits, 3)
        )

    elif ansatz == "Paper":

        params_enc = torch.nn.Parameter(
            0.01 * torch.randn(n_layers, n_qubits, 2)
        )

    return params_enc



#EN ESTE CASO, LOS PARAMETROS DEL DECODER NO LOS TOCAMOS, PERO NO HAY PARAMETROS ENCODER, ESTAN LOS VALORES DE THETA Y PHI, QUE SON LOS PARAMETROS QUE VAMOS A ENTRENAR DE LA TECNICA DE ENCODING
def inic_params_SQ(block_size, resize_dim, n_layers, opt_params):
    thetas = {}
    phis = {}
    params_decs = {}
    
    n_features = block_size * block_size  # nº de píxeles por bloque

    for i in range(0, resize_dim[0], block_size):
        for j in range(0, resize_dim[1], block_size):

            key = (i, j)

            params_decs[key] = torch.nn.Parameter(
                0.01 * torch.randn(n_layers, 2, 3)
            )

            if opt_params == False:
                thetas[key] = torch.nn.Parameter(
                    0.1 * torch.randn(n_layers, n_features)
                )
                phis[key] = torch.nn.Parameter(
                    0.1 * torch.randn(n_layers, n_features)
                )
            else:
                # ✅ ESTO DEBE IR AQUÍ
                thetas[key] = torch.nn.Parameter(
                    0.1 * torch.randn(n_layers, block_size)
                )
                phis[key] = torch.nn.Parameter(
                    0.1 * torch.randn(n_layers, block_size)
                )

    print("Theta : ", thetas)
    print("Phi : ", phis)
    print("Params decoder : ", params_decs)
    return thetas, phis, params_decs


################A REVISAR################



def crear_optimizador_SQ(optimizer_name, thetas, phis, params_decs, tasa_de_aprendizaje):
    
    all_params = (
        list(thetas.values()) +
        list(phis.values()) +
        list(params_decs.values())
    )

    if optimizer_name == "Adam":
        return torch.optim.Adam(all_params, lr=tasa_de_aprendizaje)
    elif optimizer_name == "GradientDescent":
        return torch.optim.SGD(all_params, lr=tasa_de_aprendizaje)
    else:
        raise ValueError(f"Optimizador desconocido: {optimizer_name}")
    

def inic_params_SQ_global(block_size, n_layers, n_qubits, ansatz):

    # -----------------------------
    # PARÁMETROS GLOBALES THETA/PHI
    # -----------------------------

    n_features = block_size * block_size  # nº de píxeles por bloque


    theta = torch.nn.Parameter(0.1 * torch.randn(n_features))
    phi   = torch.nn.Parameter(0.1 * torch.randn(n_features))

    # -----------------------------
    # DECODER GLOBAL
    # -----------------------------

    if ansatz == "StronglyEntangling":

        params_dec = torch.nn.Parameter(
            0.01 * torch.randn(n_layers, n_qubits, 3)
        )

    elif ansatz == "Paper":

        params_dec = torch.nn.Parameter(
            0.01 * torch.randn(n_layers, n_qubits, 2)
        )

    return theta, phi, params_dec

def crear_optimizador_SQ_orig(optimizer_name, theta, phi, params_dec, tasa_de_aprendizaje):


    if optimizer_name == "Adam":
        return torch.optim.Adam([theta, phi, params_dec], lr=tasa_de_aprendizaje)
    elif optimizer_name == "GradientDescent":
        return torch.optim.SGD([theta, phi, params_dec], lr=tasa_de_aprendizaje)
    else:
        raise ValueError(f"Optimizador desconocido: {optimizer_name}")
