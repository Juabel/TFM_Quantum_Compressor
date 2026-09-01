import pennylane as qml
import torch

def single_qubit_reuploading_embedding_orig(state, theta, phi, wire=0):

    theta = theta.squeeze()
    phi = phi.squeeze()

    for i, x in enumerate(state):

        angle = torch.pi * (theta[i] + phi[i] * x)

        if i % 2 == 0:
            qml.RY(angle, wires=wire)
        else:
            qml.RZ(angle, wires=wire)



def dense_angle_embedding(state, wire):
    state = state.flatten()

    for i, x in enumerate(state):
        angle = torch.pi * x

        if i % 2 == 0:
            qml.RX(angle, wires=wire)
        else:
            qml.RZ(angle, wires=wire)


def preparar_etiqueta(label, wires):

    bits = format(int(label), "02b")   # 0 -> 00, 1 -> 01, 2 -> 10, 3 -> 11

    for bit, wire in zip(bits, wires):
        if bit == "1":
            qml.PauliX(wires=wire)


def swap_test(register1, register2, ancilla):

    qml.Hadamard(wires=ancilla)

    for w1, w2 in zip(register1, register2):
        qml.CSWAP(wires=[ancilla, w1, w2])

    qml.Hadamard(wires=ancilla)


def create_autoencoder_recon_ampl_dagger(dev):

    @qml.qnode(dev, interface="torch")
    def autoencoder_recon_ampl(state, params_enc, denseAngle, ansatz, mejora):

        encoders = {
            "StronglyEntangling": encoder_strong,
            "Paper": encoder_paper,
        }
        encoder_fn = encoders[ansatz]
        decoder_fn = qml.adjoint(encoder_fn)


        if denseAngle == "True":
            dense_angle_embedding(state[:2], 0)
            dense_angle_embedding(state[2:], 1)
            qml.StatePrep(torch.tensor([1.0, 0.0]), wires=[2])
        else:
            qml.AmplitudeEmbedding(state, wires=range(8), normalize=True)
        # 2️⃣ Encoder
        encoder_fn(params_enc, mejora)

        # 5️⃣ Decoder sobre los 4 qubits que reconstruyen la imagen
        decoder_fn(params_enc, mejora)       
        
        # 6️⃣ Reconstrucción con las probabilidades
        if denseAngle == "True":
            output = [
                val
                for i in [0, 1]
                for val in (qml.expval(qml.PauliZ(i)), qml.expval(qml.PauliX(i)))
            ]
            # trash = qml.expval(qml.Projector([0], wires=[2]))


        else:
            output = qml.probs(wires=range(8))

                
        # ahora los qubits basura están en 2
        return output
        # return output, trash

    return autoencoder_recon_ampl


def create_classifier_ampl(dev):
    @qml.qnode(dev, interface="torch")
    def classifier_ampl(state,label,params_enc,denseAngle,ansatz,mejora):
        encoders = {
            "StronglyEntangling": encoder_strong,
            "Paper": encoder_paper,
        }

        encoder_fn = encoders[ansatz]

        qml.AmplitudeEmbedding(state, wires=range(8), normalize=True)

        encoder_fn(params_enc, mejora)

        preparar_etiqueta(label, wires=[8,9])

        swap_test(
            register1=[6,7],
            register2=[8,9],
            ancilla=10
        )

        return qml.probs(wires=10)
    return classifier_ampl





def create_encoder_probs_ampl(dev):
    @qml.qnode(dev, interface="torch")
    def encoder_probs(state , label, params, denseAngle, ansatz, mejora):

        encoders = {
            "StronglyEntangling": encoder_strong,
            "Paper": encoder_paper,
        }
        encoder_fn = encoders[ansatz]

        # Embedding
        if denseAngle == "True":
            dense_angle_embedding(state[:2], 0)
            dense_angle_embedding(state[2:], 1)
            qml.StatePrep(torch.tensor([1.0, 0.0]), wires=[2])
        else:
            qml.AmplitudeEmbedding(state, wires=range(8), normalize=True)
            preparar_etiqueta(label, wires=range(8,12))

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
            #el estado que tiene la mayor probabilidad de obtencion que seria la etiqueta al fin y al cabocon
            i = i + 1

        return output

    return encoder_probs

def loss_autoencoder_circuito_ampl(state, class_score):

    # recon_loss = torch.mean((state - pixels_recon.flatten())**2)

    # print("Valor loss basura que tiene que ir disminuyendo: ", trash_loss.item())

    # lambda_class = 0.5
    # lambda_recon = 0.5
    p0 = class_score[0]

    fidelity = 2*p0 - 1
    class_loss = 1 - fidelity

    loss = class_loss

    print(
    f"p0={p0.item():.4f} "
    f" fidelity={fidelity.item():.4f} "
    f" loss={loss.item():.4f}"
    )

    return loss






#################### FUNCIONES USADAS PARA ANGLE ENCODING ###################

def create_autoencoder_recon_dagger(dev):

    @qml.qnode(dev, interface="torch")
    def autoencoder_recon(state, params_enc, n_qubits_utiles, n_qubits_total, basis, ansatz, mejora):


        encoders = {
            "StronglyEntangling": encoder_strong,
            "Paper": encoder_paper,
        }
        encoder_fn = encoders[ansatz]
        decoder_fn = qml.adjoint(encoder_fn)


        if basis == "True":
            qml.BasisState(state, wires=[0,1,2,3])
            qml.StatePrep(torch.tensor([1.0, 0.0]), wires=[4])
            qml.StatePrep(torch.tensor([1.0, 0.0]), wires=[5])


        else:
            qml.AngleEmbedding(state * torch.pi, wires=[0,1,2,3], rotation="Y")
            qml.StatePrep(torch.tensor([1.0, 0.0]), wires=[4])
            qml.StatePrep(torch.tensor([1.0, 0.0]), wires=[5])


        # 2️⃣ Encoder
        encoder_fn(params_enc, mejora)

        # 4️⃣ SWAP con ancillas para reinicializar
        qml.SWAP(wires=[3,5])  # SWAP entre el último qubit útil y el último ancilla
        qml.SWAP(wires=[2,4])  # SWAP entre el penúltimo qubit útil y el penúltimo ancilla


        # 5️⃣ Decoder sobre los 4 qubits que reconstruyen la imagen
        decoder_fn(params_enc, mejora)       

        # 6️⃣ Reconstrucción
        recon = [qml.expval(qml.PauliZ(i)) for i in range(4)]
        
        # ahora los qubits basura están en 4 y 5
 
        # trash = qml.expval(qml.Projector([0,0], wires=[4, 5]))        

        return recon

        # return recon, trash

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
            qml.BasisState(state, wires=[0,1,2,3])
            qml.StatePrep(torch.tensor([1.0, 0.0]), wires=[4])
            qml.StatePrep(torch.tensor([1.0, 0.0]), wires=[5])

        else:
            qml.AngleEmbedding(state * torch.pi, wires=[0,1,2,3], rotation="Y")
            qml.StatePrep(torch.tensor([1.0, 0.0]), wires=[4])
            qml.StatePrep(torch.tensor([1.0, 0.0]), wires=[5])

        # Encoder entrenado
        encoder_fn(params, mejora)

        # Medimos SOLO latentes
        # if dense == "True":
        # latent = [qml.expval(qml.PauliZ(i)) for i in range(2)]
        latent = [qml.expval(qml.PauliZ(i)) for i in [0,1]]

        return latent

    return encoder_probs
    

def loss_autoencoder_circuito(block_norm, pixel_expvals):

    recon_loss = torch.mean((block_norm - pixel_expvals.flatten())**2)


    return recon_loss, recon_loss



#################### FUNCIONES USADAS PARA SINGLE CUBIT ENCODING ####################

def create_autoencoder_recon_single(dev):

    @qml.qnode(dev, interface="torch")
    def autoencoder_recon(state, theta, phi, params_dec, ansatz, mejora):

        #SOLO HAY DOS CUBITS, UNO CON INFO Y EL OTRO QUE ENTRARA EN EL DECODER COMO 0

        decoders = {
            "StronglyEntangling": encoder_strong,
            "Paper": encoder_paper,
        }
        decoder_fn = decoders[ansatz]


        # 1️⃣ Embedding
        single_qubit_reuploading_embedding_orig(state, theta, phi, wire=0)
        qml.StatePrep([1,0], wires=1)

        decoder_fn(params_dec, mejora)

        output = [
            val
            for i in [0, 1]
            for val in (qml.expval(qml.PauliZ(i)), qml.expval(qml.PauliY(i)))
        ]


        # 6️⃣ Reconstrucción
        # output = qml.probs(wires=[0,1])
        
        return output

    return autoencoder_recon

def create_encoder_probs_SQ(dev):
    @qml.qnode(dev, interface="torch")
    def encoder_probs(state, theta, phi):

        single_qubit_reuploading_embedding_orig(state, theta, phi, wire=0)

        output = [
            val
            for i in [0]
            for val in (qml.expval(qml.PauliZ(i)), qml.expval(qml.PauliY(i)))
        ]

        return output

    return encoder_probs

def loss_autoencoder_circuito_SQ(state, pixels_recon):

    recon_loss = torch.mean((state - pixels_recon)**2)

    return recon_loss


################# CIRCUITOS ANSATZ #################

def encoder_strong(params_enc, mejora):
    
    qml.StronglyEntanglingLayers(
        params_enc,
        wires=range(8)
    )

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
