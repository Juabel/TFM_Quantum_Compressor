import pennylane as qml
import torch

def single_qubit_reuploading_embedding_orig(state, theta, phi, opt_params, wire=0):

    theta = theta.flatten()
    phi = phi.flatten()
    

    for i, x in enumerate(state):
        if opt_params == True:

            theta_val = theta[0]
            phi_val = phi[0]
            angle = theta_val + phi_val * x

        else:
            angle = theta[i] + phi[i] * x

        if i % 2 == 0: 
            qml.RY(angle, wires=wire)
        else:
            qml.RZ(angle, wires=wire)


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

        else:
            output = qml.probs(wires=[0,1])
            trash = qml.expval(qml.Projector([0], wires=[2]))
                
        # ahora los qubits basura están en 2

        return output, trash
    return autoencoder_recon_ampl


def create_autoencoder_recon_ampl_dagger(dev):

    @qml.qnode(dev, interface="torch")
    def autoencoder_recon_ampl(state, params_enc, params_dec, denseAngle, ansatz, mejora):

        encoders = {
            "StronglyEntangling": encoder_strong,
            "Paper": encoder_paper,
        }
        encoder_fn = encoders[ansatz]
        decoder_fn = qml.adjoint(encoder_fn)


        if denseAngle == "True":
            dense_angle_embedding(state * torch.pi, 2)
        else:
            qml.AmplitudeEmbedding(state, wires=[0,1], normalize=True)
            # qml.AmplitudeEmbedding(state, wires=[0,1], normalize=False)

        # 2️⃣ Encoder
        encoder_fn(params_enc, mejora)


        # 4️⃣ SWAP con ancillas para reinicializar
        qml.SWAP(wires=[1,2])

        # 5️⃣ Decoder sobre los 4 qubits que reconstruyen la imagen
        decoder_fn(params_enc, mejora)       
        
        # 6️⃣ Reconstrucción con las probabilidades
        if denseAngle == "True":
            output = [
                val
                for i in [0, 1]
                for val in (qml.expval(qml.PauliZ(i)), qml.expval(qml.PauliX(i)))
            ]
            trash = qml.expval(qml.Projector([0], wires=[2]))


        else:
            output = qml.probs(wires=[0,1])
            trash = qml.expval(qml.Projector([0], wires=[2]))

                
        # ahora los qubits basura están en 2

        return output, trash

    return autoencoder_recon_ampl



def create_encoder_probs_ampl(dev):
    @qml.qnode(dev, interface="torch")
    def encoder_probs(state , params, denseAngle, ansatz, mejora):

        encoders = {
            "StronglyEntangling": encoder_strong,
            "Paper": encoder_paper,
        }
        encoder_fn = encoders[ansatz]

        # Embedding
        if denseAngle == "True":
            dense_angle_embedding(state * torch.pi, 2)
        else:
            qml.AmplitudeEmbedding(state, wires=[0,1], normalize=True)
            # qml.AmplitudeEmbedding(state, wires=[0,1], normalize=False)

        # Encoder entrenado
        encoder_fn(params, mejora)

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

def loss_autoencoder_circuito_ampl(state, pixels_recon, trash, lambda_trash):

    recon_loss = torch.mean((state - pixels_recon)**2)

    trash_loss = 1 - trash
    # print("Valor loss basura que tiene que ir disminuyendo:", trash_loss.item())
    # print("Probabilidad |00> basura:", trash.detach().cpu().numpy())
    return recon_loss + lambda_trash * trash_loss, recon_loss






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


        return recon, trash

    return autoencoder_recon

def create_autoencoder_recon_dagger(dev):

    @qml.qnode(dev, interface="torch")
    def autoencoder_recon(state, params_enc, params_dec, n_qubits_utiles, n_qubits_total, basis, ansatz, mejora):


        encoders = {
            "StronglyEntangling": encoder_strong,
            "Paper": encoder_paper,
        }
        encoder_fn = encoders[ansatz]
        decoder_fn = qml.adjoint(encoder_fn)


        if basis == "True":
            qml.BasisState(state, wires=range(4))
        else:
            qml.AngleEmbedding(state * torch.pi, wires=range(n_qubits_utiles), rotation="Y")

        # 2️⃣ Encoder
        encoder_fn(params_enc, mejora)

        # 4️⃣ SWAP con ancillas para reinicializar
        qml.SWAP(wires=[n_qubits_utiles - 1,n_qubits_total - 1])  # SWAP entre el último qubit útil y el último ancilla
        qml.SWAP(wires=[n_qubits_utiles - 2,n_qubits_total - 2])  # SWAP entre el penúltimo qubit útil y el penúltimo ancilla


        # Ahora:
        # qubits 0,1 = latentes
        # qubits 2,3 = |00>

        # 5️⃣ Decoder sobre los 4 qubits que reconstruyen la imagen
        decoder_fn(params_enc, mejora)       

        # 6️⃣ Reconstrucción
        recon = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits_utiles)]
        
        # ahora los qubits basura están en 4 y 5
 
        trash = qml.expval(qml.Projector([0,0], wires=[n_qubits_total - 2, n_qubits_total - 1]))        


        return recon, trash

    return autoencoder_recon



def create_encoder_probs(dev):
    @qml.qnode(dev, interface="torch")
    def encoder_probs(state , params,  n_qubits_utiles, n_qubits_total, basis, ansatz, mejora):


        encoders = {
            "StronglyEntangling": encoder_strong,
            "Paper": encoder_paper,
        }
        encoder_fn = encoders[ansatz]


        # Embedding
        if basis == "True":
            qml.BasisState(state, wires=range(4))
        else:
            qml.AngleEmbedding(state * torch.pi, wires=range(n_qubits_utiles), rotation="Y")

        # Encoder entrenado
        encoder_fn(params, mejora)

        # Medimos SOLO latentes
        # if dense == "True":
        latent = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits_total - n_qubits_utiles)]
        # latent = [qml.expval(qml.PauliZ(i)) for i in [0,1]]

        return latent

    return encoder_probs
    

def loss_autoencoder_circuito(block_norm, pixel_expvals, trash_pixels_expvals, lambda_trash):

    recon_loss = torch.mean((block_norm - pixel_expvals.flatten())**2)

    trash_loss = 1 - trash_pixels_expvals
    # print("Valor loss basura que tiene que ir disminuyendo:", trash_loss.item())
    # print("Probabilidad |00> basura:", trash_pixels_expvals.detach().cpu().numpy())
    # print("Penalización de Bloch:", bloch_penalty.detach().cpu().numpy())

    return recon_loss + lambda_trash * trash_loss, recon_loss



#################### FUNCIONES USADAS PARA SINGLE CUBIT ENCODING ####################

def create_autoencoder_recon_single(dev):

    @qml.qnode(dev, interface="torch")
    def autoencoder_recon(state, theta, phi, params_dec, opt_params, ansatz, mejora):

        #SOLO HAY DOS CUBITS, UNO CON INFO Y EL OTRO QUE ENTRARA EN EL DECODER COMO 0

        decoders = {
            "StronglyEntangling": encoder_strong,
            "Paper": encoder_paper,
        }
        decoder_fn = decoders[ansatz]


        # 1️⃣ Embedding
        single_qubit_reuploading_embedding_orig(state, theta, phi, opt_params, wire=0)

        decoder_fn(params_dec, mejora)

        # 6️⃣ Reconstrucción
        output = qml.probs(wires=[0,1])
        
        return output

    return autoencoder_recon

def create_encoder_probs_SQ(dev):
    @qml.qnode(dev, interface="torch")
    def encoder_probs(state, theta, phi, opt_params):

        single_qubit_reuploading_embedding_orig(state, theta, phi, opt_params, wire=0)

        output = qml.probs(wires=[0])

        return output

    return encoder_probs

def loss_autoencoder_circuito_SQ(state, pixels_recon):

    recon_loss = torch.mean((state - pixels_recon)**2)

    return recon_loss


################# CIRCUITOS ANSATZ #################

def encoder_strong(params_enc, mejora):
    
    n_layers, n_qubits, _ = params_enc.shape
    #dependiendo del numero de cubits, aplico strongly de una manera u otra

    if n_qubits == 4:
        qml.StronglyEntanglingLayers(params_enc, wires=[0,1,2,3])
    elif n_qubits == 2:
        qml.StronglyEntanglingLayers(params_enc, wires=[0,1])

def encoder_paper(params_enc, mejora):
    n_layers, n_qubits, _ = params_enc.shape

    for layer in range(n_layers):
        # RY inicial
        for q in range(n_qubits):
            qml.RY(params_enc[layer, q, 0], wires=q)

        if mejora == False:
            # Entrelazamiento tipo cadena + cierre final
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])

            # cierre del circuito: último como control, primero como target
            qml.CNOT(wires=[n_qubits - 1, 0])
        else:
            if n_qubits == 4:
                qml.CNOT([0,1])
                qml.CNOT([2,3])

                # vertical
                qml.CNOT([0,2])
                qml.CNOT([1,3])
            elif n_qubits == 2:
                qml.CNOT([0,1])

        # RY final
        for q in range(n_qubits):
            qml.RY(params_enc[layer, q, 1], wires=q)
