import pennylane as qml
import torch
import math



def single_qubit_embedding(state, params_rot, k_qubits=1, rotation_type="RY"):
    # Seleccionamos los primeros k_qubits
    qubits = list(range(k_qubits))
    
    # Dividimos el vector de entrada equitativamente
    chunk_size = int(torch.ceil(torch.tensor(len(state) / k_qubits, dtype=torch.float32)))

    # Recorrer cada qubit
    for qi, q in enumerate(qubits):
        # Extraer su segmento correspondiente
        start = qi * chunk_size
        end = min(start + chunk_size, len(state))
        angles = state[start:end]

        # Aplicar todas las rotaciones de ese segmento al mismo qubit
        for angle in angles:
            if rotation_type == "RY":
                qml.RY(angle, wires=q)
            elif rotation_type == "RX":
                qml.RX(angle, wires=q)
            elif rotation_type == "RZ":
                qml.RZ(angle, wires=q)
            else:
                raise ValueError("rotation_type must be RX, RY, or RZ")

#REVISAR SINGLE QUBIT ENCODING, ESTO NO ESTA PROBADO AUN


def dense_angle_embedding(state, k_qubits):
    # Seleccionamos los primeros k_qubits
    qubits = list(range(k_qubits))
    
    # Dividimos el vector de entrada equitativamente
    chunk_size = int(torch.ceil(torch.tensor(len(state) / k_qubits, dtype=torch.float32)))

    # Recorrer cada qubit
    for qi, q in enumerate(qubits):
        # Extraer su segmento correspondiente
        start = qi * chunk_size
        end = min(start + chunk_size, len(state))
        angles = state[start:end]

        # Aplicar rotaciones alternando RY y RZ
        for i, angle in enumerate(angles):
            if i % 2 == 0:
                qml.RY(angle, wires=q)
            else:
                qml.RZ(angle, wires=q)

#REVISAR DENSE ANGLE ENCODING, ESTO NO ESTA PROBADO AUN

def angle_embedding(state, params_rot, n_q):
    qml.AngleEmbedding(state, wires=range(n_q))

def amplitude_embedding(block_flat, params_rot, n_q):
    qml.AmplitudeEmbedding(block_flat, wires=range(n_q), normalize=True)

def pauliX (state, params_rot, n_q):
    for k in range(n_q):
        qml.PauliX(wires=k)

def pauliY (state, params_rot, n_q):
    for k in range(n_q):
        qml.PauliY(wires=k)

def hadamard_all(state, params_rot, n_q):
    for k in range(n_q):
        qml.Hadamard(wires=k)


#ESTA PUERTA ESTA DE REVISARLA 
def toffoli_gate(state, params_rot):
    qml.Toffoli(wires=[0, 1, 2])  # Control qubits: 0,1; Target qubit: 2

def rotations_RY(state, params, n_q):
    params = params.flatten()  # Asegurarse de que sea un vector plano
    if len(params) < n_q:
        raise ValueError(f"Se requieren {n_q} parámetros, pero solo hay {len(params)}")
    for k in range(n_q):
        qml.RY(params[k], wires=k)

def rotations_RY_latent(state, params, n_q):
    k_latent = math.ceil(math.log2(n_q))
    params = params.flatten()

    if len(params) < k_latent:
        raise ValueError(
            f"Se requieren {k_latent} parámetros, pero solo hay {len(params)}"
        )

    for k in range(k_latent):
        qml.RY(params[k], wires=k)

def rotations_RX(state, params, n_q):
    for k in range(n_q):
        qml.RX(params[k], wires=k)

def phase_gate(state, params, n_q):
    for k in range(n_q):
        qml.PhaseShift(params[k], wires=k)

def T_gate(state, params, n_q):
    for k in range(n_q):
        qml.T(wires=k)

def rotations_RZ(state, params, n_q):
    for k in range(n_q):
        qml.RZ(params[k], wires=k)

def CZ_gate(state, params, n_q):
    for k in range(0, n_q - 1, 2):
        qml.CZ(wires=[k, k + 1])

def CY_gate(state, params, n_q):
    for k in range(0, n_q - 1, 2):
        qml.CY(wires=[k, k + 1])

def SWAP_gate(state, params, n_q):
    for k in range(0, n_q - 1, 2):
        qml.SWAP(wires=[k, k + 1])

def cnot_layer(state, params_rot, n_q):
    for k in range(n_q - 1):
        qml.CNOT(wires=[k+1, k])
    
def cnot_layer_compress(state, params_rot, n_q):
    k_latent = math.ceil(math.log2(n_q))

    # Los últimos qubits se "pliegan" sobre los primeros k
    for src in range(k_latent, n_q):
        tgt = src % k_latent
        qml.CNOT(wires=[src, tgt])


CIRCUIT_MODULES = {
    "Amplitude": amplitude_embedding,
    "Angle": angle_embedding,
    "DenseAngle": dense_angle_embedding,
    "SingleQubit": single_qubit_embedding,
    "PauliX": pauliX,
    "PauliY": pauliY,
    "Hadamard": hadamard_all,
    "Toffoli": toffoli_gate,
    "RotacionesY": rotations_RY,
    "RotacionesY_Compress": rotations_RY_latent,
    "RotacionesX": rotations_RX,
    "RotacionesZ": rotations_RZ,
    "Phase": phase_gate,
    "SWAP": SWAP_gate,
    "T": T_gate,
    "CZ": CZ_gate,
    "CY": CY_gate,
    "CNOT": cnot_layer,
    "CNOT_Compress" : cnot_layer_compress
}


def create_autoencoder_recon_ampl(dev):

    @qml.qnode(dev, interface="torch")
    def autoencoder_recon_ampl(state, params_enc, params_dec, denseAngle):

        # 1️⃣ Embedding en los 4 qubits de datos
        if denseAngle == "True":
            dense_angle_embedding(state * torch.pi, 2)
        else:
            qml.AmplitudeEmbedding(state, wires=[0,1], normalize=True)
            # qml.AmplitudeEmbedding(state, wires=[0,1], normalize=False)

        # 2️⃣ Encoder
        qml.StronglyEntanglingLayers(params_enc, wires=[0,1])


        # 4️⃣ SWAP con ancillas para reinicializar
        qml.SWAP(wires=[1,2])

        # 5️⃣ Decoder sobre los 4 qubits que reconstruyen la imagen
        qml.StronglyEntanglingLayers(params_dec, wires=[0,1])

        # 6️⃣ Reconstrucción con la raiz cuadrada de las probabilidades, es decir, las amplitudes
        if denseAngle == "True":
            output = [
                val
                for i in [0, 1]
                for val in (qml.expval(qml.PauliZ(i)), qml.expval(qml.PauliX(i)))
            ]
            trash = qml.expval(qml.Projector([0], wires=[2]))

            # Medición de Y para penalización
            # latent_y0 = qml.expval(qml.PauliY(0))
            # latent_y1 = qml.expval(qml.PauliY(1))

            #PRUEBA devolviendo los castigo en 0
            latent_y0 = qml.expval(0 * qml.PauliY(0))
            latent_y1 = qml.expval(0 * qml.PauliY(1))
        else:
            output = qml.probs(wires=[0,1])
            trash = qml.expval(qml.Projector([0], wires=[2]))

            latent_y0 = qml.expval(0 * qml.PauliZ(0))
            latent_y1 = qml.expval(0 * qml.PauliZ(1))
                
        # ahora los qubits basura están en 2

        return output, trash, latent_y0, latent_y1

    return autoencoder_recon_ampl


def create_autoencoder_recon_ampl_dagger(dev):

    @qml.qnode(dev, interface="torch")
    def autoencoder_recon_ampl(state, params_enc, params_dec, denseAngle):

        if denseAngle == "True":
            dense_angle_embedding(state * torch.pi, 2)
        else:
            qml.AmplitudeEmbedding(state, wires=[0,1], normalize=True)
            # qml.AmplitudeEmbedding(state, wires=[0,1], normalize=False)

        # 2️⃣ Encoder
        qml.StronglyEntanglingLayers(params_enc, wires=[0,1])


        # 4️⃣ SWAP con ancillas para reinicializar
        qml.SWAP(wires=[1,2])

        # 5️⃣ Decoder sobre los 4 qubits que reconstruyen la imagen
        qml.adjoint(qml.StronglyEntanglingLayers)(params_enc, wires=[0,1])

        # 6️⃣ Reconstrucción con las probabilidades
        if denseAngle == "True":
            output = [
                val
                for i in [0, 1]
                for val in (qml.expval(qml.PauliZ(i)), qml.expval(qml.PauliX(i)))
            ]
            trash = qml.expval(qml.Projector([0], wires=[2]))

            # Medición de Y para penalización
            latent_y0 = qml.expval(qml.PauliY(0))
            latent_y1 = qml.expval(qml.PauliY(1))

            #PRUEBA devolviendo los castigo en 0
            # latent_y0 = qml.expval(0 * qml.PauliY(0))
            # latent_y1 = qml.expval(0 * qml.PauliY(1))
        else:
            output = qml.probs(wires=[0,1])
            trash = qml.expval(qml.Projector([0], wires=[2]))

            latent_y0 = qml.expval(0 * qml.PauliY(0))
            latent_y1 = qml.expval(0 * qml.PauliY(1))
                
        # ahora los qubits basura están en 2

        return output, trash, latent_y0, latent_y1

    return autoencoder_recon_ampl



def create_encoder_probs_ampl(dev):
    @qml.qnode(dev, interface="torch")
    def encoder_probs(state , params, denseAngle):

        # Embedding
        if denseAngle == "True":
            dense_angle_embedding(state * torch.pi, 2)
        else:
            qml.AmplitudeEmbedding(state, wires=[0,1], normalize=True)
            # qml.AmplitudeEmbedding(state, wires=[0,1], normalize=False)

        # Encoder entrenado
        qml.StronglyEntanglingLayers(params, wires=[0,1])

        # Medimos SOLO latentes
        if denseAngle == "True":
            output = [
                val
                for i in [0]
                for val in (qml.expval(qml.PauliZ(i)), qml.expval(qml.PauliX(i)))
            ]
        else:
            output = qml.probs(wires=[0])

        return output

    return encoder_probs

def loss_autoencoder_circuito_ampl(state, pixels_recon, trash, lambda_trash, bloch_penalty, lambda_bloch):

    recon_loss = torch.mean((state - pixels_recon)**2)

    trash_loss = 1 - trash
    # print("Valor loss basura que tiene que ir disminuyendo:", trash_loss.item())
    print("Probabilidad |00> basura:", trash.detach().cpu().numpy())
    return recon_loss + lambda_trash * trash_loss + lambda_bloch * bloch_penalty, recon_loss






#################### FUNCIONES USADAS PARA ANGLE ENCODING ####################

def create_autoencoder_recon(dev):

    @qml.qnode(dev, interface="torch")
    def autoencoder_recon(state, params_enc, params_dec, n_qubits_utiles, n_qubits_total, basis):

        # 1️⃣ Embedding en los 4 qubits de datos

        if basis == "True":
            qml.BasisState(state, wires=range(4))
        else:
            qml.AngleEmbedding(state * torch.pi, wires=range(n_qubits_utiles), rotation="Y")

        # 2️⃣ Encoder
        qml.StronglyEntanglingLayers(params_enc, wires=range(n_qubits_utiles))


        # 4️⃣ SWAP con ancillas para reinicializar
        qml.SWAP(wires = [n_qubits_utiles - 1, n_qubits_total - 1])  # SWAP entre el último qubit útil y el último ancilla
        qml.SWAP(wires = [n_qubits_utiles - 2, n_qubits_total - 2])  # SWAP entre el penúltimo qubit útil y el penúltimo ancilla

        # Ahora:
        # qubits 0,1 = latentes
        # qubits 2,3 = |00>

        # 5️⃣ Decoder sobre los 4 qubits que reconstruyen la imagen
        qml.StronglyEntanglingLayers(params_dec, wires=range(n_qubits_utiles))

        # 6️⃣ Reconstrucción
        recon = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits_utiles)]
        
        # ahora los qubits basura están en 4 y 5
        trash = qml.expval(qml.Projector([0,0], wires=[n_qubits_total - 2, n_qubits_total - 1]))

        latent_x0 = qml.expval(qml.PauliX(0))
        latent_y0 = qml.expval(qml.PauliY(0))

        latent_x1 = qml.expval(qml.PauliX(1))
        latent_y1 = qml.expval(qml.PauliY(1))

        #prueba devolviendo latentes xy 0 
        # latent_x0 = qml.expval(qml.PauliX(0) * 0)
        # latent_y0 = qml.expval(qml.PauliY(0) * 0)

        # latent_x1 = qml.expval(qml.PauliX(1) * 0)
        # latent_y1 = qml.expval(qml.PauliY(1) * 0)
        return recon, trash, latent_x0, latent_y0, latent_x1, latent_y1

    return autoencoder_recon

def create_autoencoder_recon_dagger(dev):

    @qml.qnode(dev, interface="torch")
    def autoencoder_recon(state, params_enc, params_dec, n_qubits_utiles, n_qubits_total, basis):


        if basis == "True":
            qml.BasisState(state, wires=range(4))
        else:
            qml.AngleEmbedding(state * torch.pi, wires=range(n_qubits_utiles), rotation="Y")

        # 2️⃣ Encoder
        qml.StronglyEntanglingLayers(params_enc, wires=range(n_qubits_utiles))

        # 4️⃣ SWAP con ancillas para reinicializar
        qml.SWAP(wires=[n_qubits_utiles - 1,n_qubits_total - 1])  # SWAP entre el último qubit útil y el último ancilla
        qml.SWAP(wires=[n_qubits_utiles - 2,n_qubits_total - 2])  # SWAP entre el penúltimo qubit útil y el penúltimo ancilla


        # Ahora:
        # qubits 0,1 = latentes
        # qubits 2,3 = |00>

        # 5️⃣ Decoder sobre los 4 qubits que reconstruyen la imagen
        qml.adjoint(qml.StronglyEntanglingLayers)(params_enc, wires=range(n_qubits_utiles))

        # 6️⃣ Reconstrucción
        recon = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits_utiles)]
        
        # ahora los qubits basura están en 4 y 5
 
        trash = qml.expval(qml.Projector([0,0], wires=[n_qubits_total - 2, n_qubits_total - 1]))        



        latent_x0 = qml.expval(qml.PauliX(0))
        latent_y0 = qml.expval(qml.PauliY(0))

        latent_x1 = qml.expval(qml.PauliX(1))
        latent_y1 = qml.expval(qml.PauliY(1))

        #prueba devolviendo latentes xy 0 
        # latent_x0 = qml.expval(qml.PauliX(0) * 0)
        # latent_y0 = qml.expval(qml.PauliY(0) * 0)

        # latent_x1 = qml.expval(qml.PauliX(1) * 0)
        # latent_y1 = qml.expval(qml.PauliY(1) * 0)

        return recon, trash, latent_x0, latent_y0, latent_x1, latent_y1

    return autoencoder_recon



def create_encoder_probs(dev):
    @qml.qnode(dev, interface="torch")
    def encoder_probs(state , params,  n_qubits_utiles, n_qubits_total, basis):

        # Embedding
        if basis == "True":
            qml.BasisState(state, wires=range(4))
        else:
            qml.AngleEmbedding(state * torch.pi, wires=range(n_qubits_utiles), rotation="Y")

        # Encoder entrenado
        qml.StronglyEntanglingLayers(
            params,
            wires=range(n_qubits_utiles),
        )

        # Medimos SOLO latentes
        # if dense == "True":
        latent = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits_total - n_qubits_utiles)]
        # latent = [qml.expval(qml.PauliZ(i)) for i in [0,1]]

        return latent

    return encoder_probs
    

def loss_autoencoder_circuito(block_norm, pixel_expvals, trash_pixels_expvals, lambda_trash, bloch_penalty, lambda_bloch):

    recon_loss = torch.mean((block_norm - pixel_expvals.flatten())**2)

    trash_loss = 1 - trash_pixels_expvals
    # print("Valor loss basura que tiene que ir disminuyendo:", trash_loss.item())
    print("Probabilidad |00> basura:", trash_pixels_expvals.detach().cpu().numpy())
    # print("Penalización de Bloch:", bloch_penalty.detach().cpu().numpy())

    return recon_loss + lambda_trash * trash_loss + lambda_bloch * bloch_penalty, recon_loss
