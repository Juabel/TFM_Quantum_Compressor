import pennylane as qml
import torch



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


def dense_angle_embedding(state, params_rot, k_qubits=3, rotation_type="RY"):
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
    "RotacionesX": rotations_RX,
    "RotacionesZ": rotations_RZ,
    "Phase": phase_gate,
    "SWAP": SWAP_gate,
    "T": T_gate,
    "CZ": CZ_gate,
    "CY": CY_gate,
    "CNOT": cnot_layer
}



def create_circuit_module_dec(dev_dec, n_qubits_dec, tecnica_de_decoding_ansatz):
    @qml.qnode(dev_dec, interface="torch")
    def circuit_decoder(state, params_rot):

        for module_name, num_layers in tecnica_de_decoding_ansatz.items():

            module_fn = CIRCUIT_MODULES[module_name]

            for _ in range(num_layers):
                module_fn(state, params_rot, n_qubits_dec)

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



def create_circuit_meas(dev, n_qubits, tecnica_de_encoding_ansatz):
    @qml.qnode(dev, interface="torch")
    def circuit_meas(state, params_rot):
        for module_name, num_layers in tecnica_de_encoding_ansatz.items():

            module_fn = CIRCUIT_MODULES[module_name]

            for _ in range(num_layers):
                module_fn(state, params_rot, n_qubits)

        return [qml.expval(qml.PauliZ(k)) for k in range(n_qubits)]

    return circuit_meas


#def decoder_state(compressed_features, params_dec):
#    for module_name, num_layers in tecnica_de_decoding_ansatz.items():
#
 #       module_fn = CIRCUIT_MODULES[module_name]
#
 #       for _ in range(num_layers):
  #          module_fn(compressed_features, params_dec, n_qubits_dec)
#
 #   return qml.state()