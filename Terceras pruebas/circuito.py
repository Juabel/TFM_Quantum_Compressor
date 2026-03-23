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


def loss_autoencoder_amplitude(z_vals, block_norm):

    loss = qml.math.mean((block_norm - z_vals) ** 2)

    return loss


def create_circuit_module_dec(dev_dec, n_qubits_dec):
    @qml.qnode(dev_dec, interface="torch", diff_method="backprop")
    def circuit_decoder(state, params_rot):

        angle_embedding(state, params_rot, n_qubits_dec)
        qml.StronglyEntanglingLayers(params_rot, wires=range(n_qubits_dec))


        # for module_name, num_layers in tecnica_de_decoding_ansatz.items():

        #     module_fn = CIRCUIT_MODULES[module_name]

        #     for _ in range(num_layers):
        #         module_fn(state, params_rot, n_qubits_dec)

        # Devolver directamente el vector de probs (16 valores)
        # return qml.probs(wires=range(n_qubits_dec))
        
        # --- Medidas personalizadas ---
        measurements = []

        # Para cada qubit simple: Pauli-Z, Pauli-X, Pauli-Y
        for q in range(n_qubits_dec):
            # Z
            measurements.append(qml.expval(qml.PauliZ(q)))
            # X
            measurements.append(qml.expval(qml.PauliX(q)))

        # --- Mediciones combinadas (ejemplo: últimos 2 qubits)
        if n_qubits_dec > 2:
            for q in range(n_qubits_dec):
                # Y
                measurements.append(qml.expval(qml.PauliY(q)))
            measurements.append(qml.expval(qml.PauliZ(n_qubits_dec-2) @ qml.PauliZ(n_qubits_dec-1)))
            measurements.append(qml.expval(qml.PauliX(n_qubits_dec-2) @ qml.PauliX(n_qubits_dec-1)))
            measurements.append(qml.expval(qml.PauliY(n_qubits_dec-2) @ qml.PauliY(n_qubits_dec-1)))
            measurements.append(qml.expval(qml.PauliZ(n_qubits_dec-3) @ qml.PauliZ(n_qubits_dec-1)))
        # Devolver todos concatenados como tensor
        return measurements
    return circuit_decoder



def create_circuit_meas(dev, n_qubits):
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit_meas(state, params_rot):

        #PROBAR ESTO CON AMPLITUDE Y STRONGLYENTANGLEDLAYERS

        amplitude_embedding(state, params_rot, n_qubits)
        qml.StronglyEntanglingLayers(params_rot, wires=range(n_qubits))

        # for module_name, num_layers in tecnica_de_encoding_ansatz.items():

        #     module_fn = CIRCUIT_MODULES[module_name]

        #     for _ in range(num_layers):
        #         module_fn(state, params_rot, n_qubits)

        return [qml.expval(qml.PauliZ(k)) for k in range(n_qubits)]

    return circuit_meas


def create_autoencoder_recon_ampl(dev):

    @qml.qnode(dev, interface="torch")
    def autoencoder_recon_ampl(state, params_enc, params_dec):

        # 1️⃣ Embedding en los 4 qubits de datos
        qml.AmplitudeEmbedding(state, wires=[0,1], normalize=True)
        # qml.AmplitudeEmbedding(state, wires=[0,1], normalize=False)
        # 2️⃣ Encoder
        qml.StronglyEntanglingLayers(params_enc, wires=[0,1])


        # 4️⃣ SWAP con ancillas para reinicializar
        qml.SWAP(wires=[1,2])

        # 5️⃣ Decoder sobre los 4 qubits que reconstruyen la imagen
        qml.StronglyEntanglingLayers(params_dec, wires=[0,1])

        # 6️⃣ Reconstrucción con la raiz cuadrada de las probabilidades, es decir, las amplitudes
        probs = qml.probs(wires=[0,1])
        
        # ahora los qubits basura están en 4 y 5
        trash = qml.expval(qml.Projector([0], wires=[2]))

        return probs, trash

    return autoencoder_recon_ampl


def create_autoencoder_recon_ampl_dagger(dev):

    @qml.qnode(dev, interface="torch")
    def autoencoder_recon_ampl(state, params_enc, params_dec):

        # 1️⃣ Embedding en los 4 qubits de datos
        qml.AmplitudeEmbedding(state, wires=[0,1], normalize=True)
        # qml.AmplitudeEmbedding(state, wires=[0,1], normalize=False)
        # 2️⃣ Encoder
        qml.StronglyEntanglingLayers(params_enc, wires=[0,1])


        # 4️⃣ SWAP con ancillas para reinicializar
        qml.SWAP(wires=[1,2])

        # 5️⃣ Decoder sobre los 4 qubits que reconstruyen la imagen
        qml.adjoint(qml.StronglyEntanglingLayers)(params_enc, wires=[0,1])

        # 6️⃣ Reconstrucción con las probabilidades
        probs = qml.probs(wires=[0,1])
        
        # ahora los qubits basura están en 2
        trash = qml.expval(qml.Projector([0], wires=[2]))

        return probs, trash

    return autoencoder_recon_ampl



def create_encoder_probs_ampl(dev):
    @qml.qnode(dev, interface="torch")
    def encoder_probs(state , params):

        # Embedding
        qml.AmplitudeEmbedding(state, wires=[0,1], normalize=True)
        # qml.AmplitudeEmbedding(state, wires=[0,1], normalize=False)

        # Encoder entrenado
        qml.StronglyEntanglingLayers(params, wires=[0,1])

        # Medimos SOLO latentes
        probs = qml.probs(wires=[0])


        return probs

    return encoder_probs

def loss_autoencoder_circuito_ampl(state, pixels_recon, trash, lambda_trash):

    recon_loss = torch.mean((state - pixels_recon)**2)

    trash_loss = 1 - trash
    # print("Valor loss basura que tiene que ir disminuyendo:", trash_loss.item())
    print("Probabilidad |00> basura:", trash.detach().cpu().numpy())
    return recon_loss + lambda_trash * trash_loss, recon_loss






#################### FUNCIONES USADAS PARA ANGLE ENCODING ####################

def create_autoencoder_recon(dev):

    @qml.qnode(dev, interface="torch")
    def autoencoder_recon(state, params_enc, params_dec):

        # 1️⃣ Embedding en los 4 qubits de datos
        qml.AngleEmbedding(state * torch.pi, wires=[0,1,2,3], rotation="Y")

        # 2️⃣ Encoder
        qml.StronglyEntanglingLayers(params_enc, wires=[0,1,2,3])


        # 4️⃣ SWAP con ancillas para reinicializar
        qml.SWAP(wires=[2,4])
        qml.SWAP(wires=[3,5])

        # Ahora:
        # qubits 0,1 = latentes
        # qubits 2,3 = |00>

        # 5️⃣ Decoder sobre los 4 qubits que reconstruyen la imagen
        qml.StronglyEntanglingLayers(params_dec, wires=[0,1,2,3])

        # 6️⃣ Reconstrucción
        recon = [qml.expval(qml.PauliZ(i)) for i in [0,1,2,3]]
        
        # ahora los qubits basura están en 4 y 5
        trash = qml.expval(qml.Projector([0,0], wires=[4,5]))

        latent_x0 = qml.expval(qml.PauliX(0))
        latent_y0 = qml.expval(qml.PauliY(0))

        latent_x1 = qml.expval(qml.PauliX(1))
        latent_y1 = qml.expval(qml.PauliY(1))

        return recon, trash, latent_x0, latent_y0, latent_x1, latent_y1

    return autoencoder_recon

def create_autoencoder_recon_dagger(dev):

    @qml.qnode(dev, interface="torch")
    def autoencoder_recon(state, params_enc, params_dec):

        qml.AngleEmbedding(state * torch.pi, wires=[0,1,2,3], rotation="Y")

        # 2️⃣ Encoder
        qml.StronglyEntanglingLayers(params_enc, wires=[0,1,2,3])


        # 4️⃣ SWAP con ancillas para reinicializar
        qml.SWAP(wires=[2,4])
        qml.SWAP(wires=[3,5])

        # Ahora:
        # qubits 0,1 = latentes
        # qubits 2,3 = |00>

        # 5️⃣ Decoder sobre los 4 qubits que reconstruyen la imagen
        qml.adjoint(qml.StronglyEntanglingLayers)(params_enc, wires=[0,1,2,3])

        # 6️⃣ Reconstrucción
        recon = [qml.expval(qml.PauliZ(i)) for i in [0,1,2,3]]
        
        # ahora los qubits basura están en 4 y 5
        trash = qml.expval(qml.Projector([0,0], wires=[4,5]))

        latent_x0 = qml.expval(qml.PauliX(0))
        latent_y0 = qml.expval(qml.PauliY(0))

        latent_x1 = qml.expval(qml.PauliX(1))
        latent_y1 = qml.expval(qml.PauliY(1))

        return recon, trash, latent_x0, latent_y0, latent_x1, latent_y1

    return autoencoder_recon



def create_encoder_probs(dev):
    @qml.qnode(dev, interface="torch")
    def encoder_probs(state , params):

        # Embedding
        qml.AngleEmbedding(
            state * torch.pi,
            wires=[0,1,2,3],
            rotation="Y"
        )

        # Encoder entrenado
        qml.StronglyEntanglingLayers(
            params,
            wires=[0,1,2,3],
        )

        # Medimos SOLO latentes
        latent = [qml.expval(qml.PauliZ(i)) for i in [0,1]]

        return latent

    return encoder_probs
    

def loss_autoencoder_circuito(block_norm, pixel_expvals, trash_pixels_expvals, lambda_trash, bloch_penalty, lambda_bloch):

    recon_loss = torch.mean((block_norm - pixel_expvals.flatten())**2)

    trash_loss = 1 - trash_pixels_expvals
    # print("Valor loss basura que tiene que ir disminuyendo:", trash_loss.item())
    # print("Probabilidad |00> basura:", trash_pixels_expvals.detach().cpu().numpy())
    # print("Penalización de Bloch:", bloch_penalty.detach().cpu().numpy())

    return recon_loss + lambda_trash * trash_loss + lambda_bloch * bloch_penalty, recon_loss
