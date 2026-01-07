from multiprocessing import Pool
import pennylane as qml
from pennylane import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
import glob

def run_experiment(config):


        # ------------------ Configuración variables iniciales ------------------

    dataset = "Naturales y Artificiales" #Es para el log
    num_de_imagenes = 100 #Es para el log
    files = glob.glob("SAR_Dataset/*/*.png")


    img_save_path = "PruebasTFM/400x400-200x200-4q-Amp-X-CNOT/original_resized.png" # Ruta para guardar la imagen redimensionada en base a la original
    compressed_save_path = "PruebasTFM/400x400-200x200-4q-Amp-X-CNOT/compressed_output.png" # Ruta para guardar la imagen comprimida
    ruta_log = "PruebasTFM/400x400-200x200-4q-Amp-X-CNOT/log.txt" # Ruta del archivo de log con resultados

    tasa_de_aprendizaje = 0.05 # Tasa de aprendizaje para el optimizador

    resize_dim = (200, 200) # Dimensiones para redimensionar la imagen original
    compressed_dim = (100, 100) # Dimensiones de la imagen comprimida
    block_size = 4 # Tamaño de bloque para dividir la imagen, por ejemplo, bloques de 4x4 píxeles
    # Número de bloques por lado
    num_blocks_row = resize_dim[0] // block_size   # = 20 // 4 = 5
    num_blocks_col = resize_dim[1] // block_size   # = 20 // 4 = 5
    output_block_size = 2   # Porque tiene 4 Z-vals → 2×2 image
    compressed_height = num_blocks_row * output_block_size   # 5 × 2 = 10
    compressed_width  = num_blocks_col * output_block_size   # 5 × 2 = 10
    n_qubits = 4 # Número de qubits para representar cada bloque, tiene que ser consistente con el tamaño del bloque (2^n_qubits = block_size*block_size)


    log_ratios = [] # Lista para almacenar los ratios de compresión de cada imagen procesada
    log_tamaño_original = [] # Lista para almacenar los tamaños originales de las imágenes
    log_tamaño_comprimido = [] # Lista para almacenar los tamaños comprimidos de las imágenes

    num_iteraciones_global = 20 # Número de iteraciones globales para el entrenamiento, es decir, cuántas veces se optimizan todos los bloques de la imagen
    num_iteraciones_bloque = 100 # Número de iteraciones locales para optimizar cada bloque individualmente


    optimizer_name = "Adam" # Nombre del optimizador a usar

    tecnica_de_encoding_ansatz = {
        "Amplitude": 1,
        "PauliX": 1,
        "CNOT": 1
    } # Técnica de encoding a usar en el circuito cuántico, puede ser una lista de técnicas para aplicar secuencialmente

    requires_training = any(gate in tecnica_de_encoding_ansatz for gate in ["Rotaciones"]) # Determinar si se requieren parámetros entrenables basados en la técnica de encoding

    tecnica_de_decoding_ansatz = {
        "Angle": 1, 
        "PauliX": 1, 
        "CNOT":1}

    mse_por_bloque = [] # Lista para almacenar los MSE por bloque durante la reconstrucción
    mse_global = []

    # Modo de normalización: "orig" = norma por vector (actual),
    # "scale01" = escalar píxeles a [0,1] y luego normalizar,
    # "zero_mean" = restar media y luego normalizar
    normalization_mode = "orig"  # cambiar a "orig", "scale01" o "zero_mean" según se quiera probar
    # ------------------ Variables para el log ------------------

    no_decoder = True
    fidelidades_imagen = []

    log_fidelidades_total = [] # Lista para almacenar las fidelidades medias por iteración global para graficar al final y escribir en el log

    start_time = time.time() # Tiempo de inicio para medir el tiempo total de ejecución

        # Aquí pegas tu código completo,
    # pero reemplazando las variables globales por los valores del config
    # Ejemplo:

    resize_dim = config["resize_dim"]
    compressed_dim = config["compressed_dim"]
    num_de_imagenes = config["num_de_imagenes"]
    dataset = config["dataset"]
    files = config["files"]
    seed = config["seed"]

    np.random.seed(seed)


    # ------------------ Cargar imagen ------------------


    # ------------------ Definir circuito ------------------
    dev = qml.device("default.qubit", wires=n_qubits) # Dispositivo cuántico simulado (el de por defecto)
    dev_dec = qml.device("default.qubit", wires=n_qubits) # Dispositivo cuántico simulado para el decoder

    def single_qubit_embedding(state, params_rot, k_qubits=1, rotation_type="RY"):
        # Seleccionamos los primeros k_qubits
        qubits = list(range(k_qubits))
        
        # Dividimos el vector de entrada equitativamente
        chunk_size = int(np.ceil(len(state) / k_qubits))

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
        chunk_size = int(np.ceil(len(state) / k_qubits))

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

    def angle_embedding(state, params_rot):
        qml.AngleEmbedding(state, wires=range(n_qubits))

    def amplitude_embedding(state, params_rot):
        qml.AmplitudeEmbedding(state, wires=range(n_qubits), normalize=True)

    def pauliX (state, params_rot):
        for k in range(n_qubits):
            qml.PauliX(wires=k)

    def pauliY (state, params_rot):
        for k in range(n_qubits):
            qml.PauliY(wires=k)

    def hadamard_all(state, params_rot):
        for k in range(n_qubits):
            qml.Hadamard(wires=k)

    def toffoli_gate(state, params_rot):
        if n_qubits < 3:
            raise ValueError("Toffoli gate requires at least 3 qubits")
        qml.Toffoli(wires=[0, 1, 2])  # Control qubits: 0,1; Target qubit: 2

    def rotations_RY(state, params):
        for k in range(n_qubits):
            qml.RY(params[k], wires=k)

    def rotations_RX(state, params):
        for k in range(n_qubits):
            qml.RX(params[k], wires=k)

    def phase_gate(state, params):
        for k in range(n_qubits):
            qml.PhaseShift(params[k], wires=k)

    def T_gate(state, params):
        for k in range(n_qubits):
            qml.T(wires=k)

    def rotations_RZ(state, params):
        for k in range(n_qubits):
            qml.RZ(params[k], wires=k)

    def CZ_gate(state, params):
        for k in range(0, n_qubits - 1, 2):
            qml.CZ(wires=[k, k + 1])

    def CY_gate(state, params):
        for k in range(0, n_qubits - 1, 2):
            qml.CY(wires=[k, k + 1])

    def SWAP_gate(state, params):
        for k in range(0, n_qubits - 1, 2):
            qml.SWAP(wires=[k, k + 1])

    def cnot_layer(state, params_rot):
        for k in range(n_qubits - 1):
            qml.CNOT(wires=[k, k+1])


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
        "CNOT": cnot_layer,
    }


    @qml.qnode(dev) # Definición del circuito cuántico
    def circuit(state, params_rot):
        for module_name, num_layers in tecnica_de_encoding_ansatz.items():

            module_fn = CIRCUIT_MODULES[module_name]

            for _ in range(num_layers):
                module_fn(state, params_rot)

        return qml.state()


    @qml.qnode(dev)
    def circuit_meas(state, params_rot):
        for module_name, num_layers in tecnica_de_encoding_ansatz.items():

            module_fn = CIRCUIT_MODULES[module_name]

            for _ in range(num_layers):
                module_fn(state, params_rot)

        return [qml.expval(qml.PauliZ(k)) for k in range(n_qubits)]

    @qml.qnode(dev_dec)   
    def decoder_state(compressed_features, params_dec):
        for module_name, num_layers in tecnica_de_decoding_ansatz:

            module_fn = CIRCUIT_MODULES[module_name]

            for _ in range(num_layers):
                module_fn(compressed_features, params_dec)

        return qml.state()


    def loss_decoder(zvals, params_dec, block_flat, block_mean, norm_fact):
        estado_reconstruido = decoder_state(zvals, params_dec)
        amps = np.abs(estado_reconstruido)      
        block_hat = amps * norm_fact + block_mean
        mse = qml.math.mean((block_flat - block_hat)**2)
        return mse
    # ------------------ Inicializar parámetros por bloque ------------------
    params_rot = []

    params_dec = {}
    for i in range(0, resize_dim[0], block_size): # Iterar sobre la imagen en pasos del tamaño del bloque (filas)
        for j in range(0, resize_dim[1], block_size): # Iterar sobre la imagen en pasos del tamaño del bloque (columnas)
            params_dec[(i,j)] = qml.numpy.array( np.random.uniform(0, 2*np.pi, size=n_qubits),requires_grad=True) # Inicializar parámetros aleatorios para cada bloque (este valor se optimizará luego)

    if not requires_training:
        print("No se han definido rotaciones en el circuito, por lo que no hay parámetros que optimizar.")
    else:
        print("Se han definido rotaciones en el circuito, se optimizarán los parámetros correspondientes.")
        params_por_bloque = {} # Diccionario para almacenar los parámetros de rotación por bloque
        for i in range(0, resize_dim[0], block_size): # Iterar sobre la imagen en pasos del tamaño del bloque (filas)
            for j in range(0, resize_dim[1], block_size): # Iterar sobre la imagen en pasos del tamaño del bloque (columnas)
                params_por_bloque[(i,j)] = qml.numpy.array( np.random.uniform(0, 2*np.pi, size=n_qubits),requires_grad=True) # Inicializar parámetros aleatorios para cada bloque (este valor se optimizará luego)
                                                                                                        
    # ------------------ Entrenamiento global ------------------



    fidelidades_blocks = [] # Lista para almacenar las fidelidades de cada bloque en esta iteración global

    if not requires_training:
        print("No se han definido rotaciones en el circuito, por lo que no hay parámetros que optimizar. Se omite la fase de entrenamiento.")
    else:
        for f in files:
            img = Image.open(f).resize(resize_dim) # Escala de grises (aunque ya lo este) y redimensionar
            img_array = np.array(img)
            for it_global in range(num_iteraciones_global): # Iterar sobre el número de iteraciones globales

                for i in range(0, resize_dim[0], block_size): # Iterar sobre la imagen en pasos del tamaño del bloque (filas)
                    for j in range(0, resize_dim[1], block_size): # Iterar sobre la imagen en pasos del tamaño del bloque (columnas)
                        block = img_array[i:i+block_size, j:j+block_size] # Extraer el bloque actual de la imagen
                        block_flat = block.flatten(order='F') # Aplanar el bloque a un vector (en orden Fortran para que coincida con AmplitudeEmbedding)
                        # Preparar vector según el modo de normalización elegido
                        block_flat = block_flat.astype(float)
                        block_mean = np.mean(block_flat)
                        if normalization_mode == "orig":
                            norm_fact = np.linalg.norm(block_flat)
                            if norm_fact == 0:
                                state = np.zeros_like(block_flat)
                                state[0] = 1.0
                            else:
                                state = block_flat / norm_fact
                        elif normalization_mode == "scale01":
                            # Escalar a [0,1], normalizar y mantener norm_fact en escala original (multiplicando por 255)
                            scaled = block_flat / 255.0
                            norm_scaled = np.linalg.norm(scaled)
                            if norm_scaled == 0:
                                state = np.zeros_like(scaled)
                                state[0] = 1.0
                                norm_fact = 0.0
                            else:
                                state = scaled / norm_scaled
                                # norm_fact en unidades de píxel original para reconstrucción
                                norm_fact = norm_scaled * 255.0
                        elif normalization_mode == "zero_mean":
                            centered = block_flat - np.mean(block_flat)
                            norm_centered = np.linalg.norm(centered)
                            if norm_centered == 0:
                                state = np.zeros_like(centered)
                                state[0] = 1.0
                            else:
                                state = centered / norm_centered
                                norm_fact = norm_centered
                        else:
                            raise ValueError(f"Unknown normalization_mode: {normalization_mode}")

                        params_rot = params_por_bloque[(i,j)] # Obtener los parámetros actuales para este bloque
                        if optimizer_name == "Adam":
                            opt_block = qml.AdamOptimizer(stepsize=tasa_de_aprendizaje) # Optimizer para la mini-optimización del bloque
                        elif optimizer_name == "GradientDescent":
                            opt_block = qml.GradientDescentOptimizer(stepsize=tasa_de_aprendizaje) # Optimizer para la mini-optimización del bloque

                        # Mini-optimización local del bloque
                        for it_block in range(num_iteraciones_bloque): # Iterar sobre el número de iteraciones locales para este bloque
                            print(f"Optimizando bloque ({i},{j}) - Iteración global {it_global+1}/{num_iteraciones_global} - Iteración bloque {it_block+1}/{num_iteraciones_bloque}")
                            def loss_block(p): # Función de pérdida para el bloque (negativo de la fidelidad ya que queremos maximizar la fidelidad y el optimizador minimiza)
                                final_state = circuit(state, p) # Ejecutar el circuito con los parámetros actuales
                                overlap = qml.math.vdot(state, final_state) # Calcular el solapamiento entre el estado inicial y el final
                                return -qml.math.abs(overlap)**2 # Devolver el negativo del cuadrado del valor absoluto del solapamiento (fidelidad negativa basicamente)
                            params_rot = opt_block.step(loss_block, params_rot) # Actualizar los parámetros usando el optimizador

                        
                        params_por_bloque[(i,j)] = params_rot # Guardar parámetros optimizados por bloque

                        # Evaluar fidelidad final del bloque
                        final_state = circuit(state, params_rot) # Ejecutar el circuito con los parámetros optimizados
                        overlap = qml.math.vdot(state, final_state) # Calcular el solapamiento entre el estado inicial y el final
                        fidelidad = qml.math.abs(overlap)**2 # Calcular la fidelidad final del bloque
                        fidelidades_blocks.append(float(fidelidad)) # Almacenar la fidelidad del bloque

                media_fidelidad = np.mean(fidelidades_blocks) # Calcular la fidelidad media de todos los bloques en esta iteración global (esta sera nuestra fidelidad de la imagen comprimida)
                print(f"Iteración global {it_global+1}/{num_iteraciones_global} - Fidelidad media = {media_fidelidad:.6f}") # Imprimir el progreso
                with open(ruta_log, "a") as f:
                    f.write(f"Iteracion global {it_global+1}/{num_iteraciones_global} - Fidelidad media = {media_fidelidad:.6f}\n") # Escribir en el log
            log_fidelidades_total.append(media_fidelidad) # Almacenar la fidelidad media para graficar al final
            



    # ------------------ Reconstrucción final de la imagen comprimida ------------------


    num_imagen = 0
    for f in files:
        num_imagen += 1
        print(f"Reconstruyendo imagen comprimida {num_imagen}/{len(files)}...")
        compressed_blocks = [] # Lista para almacenar los valores comprimidos de cada bloque
        img = Image.open(f).resize(resize_dim) # Redimensionar
        img_array = np.array(img)
        for i in range(0, resize_dim[0], block_size): # Iterar sobre la imagen en pasos del tamaño del bloque (filas)
            for j in range(0, resize_dim[1], block_size): # Iterar sobre la imagen en pasos del tamaño del bloque (columnas)
                block = img_array[i:i+block_size, j:j+block_size] # Extraer el bloque actual de la imagen
                block_flat = block.flatten(order='F') # Aplanar el bloque a un vector (en orden Fortran para que coincida con AmplitudeEmbedding)
                block_flat = block_flat.astype(float)
                block_mean = np.mean(block_flat) # brillo medio en unidades originales (0-255)
                # Calcular estado y norm_fact según el modo elegido (mismo comportamiento que en la fase de entrenamiento)
                if normalization_mode == "orig":
                    norm_fact = np.linalg.norm(block_flat)
                    if norm_fact == 0:
                        state = np.zeros_like(block_flat)
                        state[0] = 1.0
                    else:
                        state = block_flat / norm_fact
                elif normalization_mode == "scale01":
                    scaled = block_flat / 255.0
                    norm_scaled = np.linalg.norm(scaled)
                    if norm_scaled == 0:
                        state = np.zeros_like(scaled)
                        state[0] = 1.0
                        norm_fact = 0.0
                    else:
                        state = scaled / norm_scaled
                        norm_fact = norm_scaled * 255.0
                elif normalization_mode == "zero_mean":
                    centered = block_flat - np.mean(block_flat)
                    norm_centered = np.linalg.norm(centered)
                    if norm_centered == 0:
                        state = np.zeros_like(centered)
                        state[0] = 1.0
                    else:
                        state = centered / norm_centered
                        norm_fact = norm_centered
                else:
                    raise ValueError(f"Unknown normalization_mode: {normalization_mode}")


                # Lista para guadar todos los estados iniciales
                if requires_training:
                    params_rot = params_por_bloque[(i,j)] # Obtener los parámetros optimizados para este bloque

                #print(qml.draw(circuit)(state, params_rot)) 
                #Dibujar el circuito 

                final_state = circuit(state, params_rot) # Ejecutar el circuito con los parámetros optimizados
                overlap = qml.math.vdot(state, final_state) # Calcular el solapamiento entre el estado inicial y el final
                fidelidad = qml.math.abs(overlap)**2 # Calcular la fidelidad final del bloque
                fidelidades_blocks.append(float(fidelidad)) # Almacenar la fidelidad del bloque

                z_vals = circuit_meas(state, params_rot) # Ejecutar el circuito con los parámetros optimizados

                compressed_blocks.append(z_vals) # Almacenar el valor comprimido del bloque

                #Decoder
                if no_decoder == False:
                    if optimizer_name == "Adam":
                        dec_opt = qml.AdamOptimizer(stepsize=tasa_de_aprendizaje) # Optimizer para la mini-optimización del bloque
                    elif optimizer_name == "GradientDescent":
                        dec_opt = qml.GradientDescentOptimizer(stepsize=tasa_de_aprendizaje) # Optimizer para la mini-optimización del bloque
                    for it_dec in range(num_iteraciones_bloque): # Iterar sobre el número de iteraciones para optimizar el decoder
                        print(f"Optimizando decoder bloque ({i},{j}) - Iteración {it_dec+1}/{num_iteraciones_bloque}")
                        def loss_dec_block(p): # Función de pérdida para el decoder del bloque (MSE entre el bloque original y el reconstruido)
                            return loss_decoder(z_vals, p, block_flat, block_mean, norm_fact)
                        params_dec[(i,j)] = dec_opt.step(loss_dec_block, params_dec[(i,j)]) # Actualizar los parámetros del decoder usando el optimizador
                
                
                    mse = loss_decoder(z_vals, params_dec[(i,j)], block_flat, block_mean, norm_fact) # Calcular el MSE final del bloque reconstruido
                    mse_por_bloque.append(float(mse)) # Almacenar el MSE del bloque

        print(np.mean(fidelidades_blocks))
        fidelidades_imagen.append(np.mean(fidelidades_blocks))
        fidelidades_blocks =[]

        if no_decoder == False:
            mse_global_img = np.mean(mse_por_bloque)
            mse_global.append(mse_global_img)
            mse_por_bloque = [] # Reiniciar para la siguiente imagen
        

        compressed_img_small = np.zeros((compressed_height, compressed_width)) # Imagen comprimida inicializada en ceros

        idx = 0
        for bi in range(num_blocks_row):
            for bj in range(num_blocks_col):

                # Los 4 valores Pauli-Z de este bloque
                z0, z1, z2, z3 = compressed_blocks[idx]  # cada uno está en [-1, 1]

                # Escalar de [-1,1] a [0,255]
                z0 = int((z0 + 1) * 127.5)
                z1 = int((z1 + 1) * 127.5)
                z2 = int((z2 + 1) * 127.5)
                z3 = int((z3 + 1) * 127.5)

                # Guardarlos como 2×2 píxeles
                compressed_img_small[bi*2:(bi+1)*2,
                                    bj*2:(bj+1)*2] = np.array([
                                        [z0, z1],
                                        [z2, z3]
                                    ])

                idx += 1

        

        # Guardar las imágenes resultantes

        Image.fromarray(compressed_img_small.astype(np.uint8)).save(compressed_save_path)
        Image.fromarray(img_array.astype(np.uint8)).save(img_save_path) # Guardar la imagen redimensionada original

        # ------------------ Visualización comparativa de ambas imágenes ------------------
        #plt.subplot(1, 2, 1)
        #plt.imshow(img_array, cmap="gray")
        #plt.title(f"Original {resize_dim[0]}x{resize_dim[1]}")
        #plt.subplot(1, 2, 2)
        #plt.imshow(compressed_img_small, cmap="gray")
        #plt.title(f"Comprimida {compressed_dim[0]}x{compressed_dim[1]}")
        #plt.show()

        # ------------------ Gráfica de fidelidades totales ------------------
        #plt.figure(figsize=(10, 5))
        #plt.plot(log_fidelidades_total, color='blue')
        #plt.xlabel("Iteración")
        #plt.ylabel("Fidelidad")
        #plt.title("Evolución de la fidelidad durante todo el entrenamiento")
        #plt.grid(True)
        #plt.show()

        inicial_original_disk_size = os.path.getsize(f) / (1024 ** 2)
        original_disk_size = os.path.getsize(img_save_path) / (1024 ** 2)
        compressed_disk_size = os.path.getsize(compressed_save_path) / (1024 ** 2)
        disk_ratio = original_disk_size / compressed_disk_size if compressed_disk_size > 0 else float('inf')

        # Guardar ratio de compresión de cada imagen
        log_tamaño_original.append(original_disk_size)
        log_tamaño_comprimido.append(compressed_disk_size)
        log_ratios.append(disk_ratio)
        print(f"Imagen {num_imagen}: Tamano original = {original_disk_size:.6f} MB, Tamaño comprimido = {compressed_disk_size:.6f} MB, Ratio de compresion = {disk_ratio:.2f}x")
        
    # ------------------ Métricas ------------------


    if no_decoder == False:
        # Media de MSE global
        media = np.mean(mse_global)

    # Media de ratios de compresión
    average_disk_ratio = np.mean(log_ratios) if log_ratios else 0
    average_tamaño_original = np.mean(log_tamaño_original) if log_tamaño_original else 0
    average_tamaño_comprimido = np.mean(log_tamaño_comprimido) if log_tamaño_comprimido else 0

    fidelidad_media_result = np.mean(log_fidelidades_total)
    if no_decoder == False:
        with open (ruta_log, "a") as f:
            f.write(f"\nMSE medio global de todas las imagenes: {media:.6f}\n")
    if requires_training:
        with open(ruta_log, "a") as f:
            f.write(f"\nFidelidad media final de las imagenes comprimidas: {fidelidad_media_result:.6f}\n")


    end_time = time.time() # Tiempo de fin para medir el tiempo total de ejecución
    tiempo_total = end_time - start_time
    print(f"Tiempo total: {tiempo_total:.2f} segundos")
    fidelidad_log = np.mean(fidelidades_imagen)
    with open(ruta_log, "a") as f:
        f.write("\n\n\n")
        f.write(f"------------- HIPERPARAMETROS -------------\n")
        f.write(f"Resize dimension: {resize_dim}\n")
        f.write(f"Compressed dimension: {compressed_dim}\n")
        f.write(f"Tamano de bloques: {block_size}x{block_size}\n")
        f.write(f"Numero de qubits: {n_qubits}\n")
        f.write(f"Tecnica de encoding y de ansatz: {tecnica_de_encoding_ansatz}\n")
        f.write(f"Tipo de imagenes: {dataset}\n")
        f.write(f"Numero de imagenes: {num_de_imagenes}\n")
        f.write(f"-------------  -------------\n")
        f.write(f"Tiempo de ejecucion: {tiempo_total:.2f} segundos\n")
        f.write(f"Fidelidad completa media: {fidelidad_log:.6f}\n")
        f.write(f"Ratio de compresion: {average_disk_ratio:.2f}\n")
        f.write(f"Tamano original medio: {average_tamaño_original:.6f} MB\n")
        f.write(f"Tamano comprimido medio: {average_tamaño_comprimido:.6f} MB\n")


if __name__ == "__main__":

    configs = [
        {"resize_dim": (400,400), "compressed_dim": (200,200), "num_de_imagenes": 50, "dataset": "Naturales", "files": glob.glob("SAR_Dataset/1/*.png"), "seed": 42},
        {"resize_dim": (400,400), "compressed_dim": (200,200), "num_de_imagenes": 100, "dataset": "Naturales y Artificiales", "files": glob.glob("SAR_Dataset/*/*.png"), "seed": 123},
    ]

    with Pool(processes=4) as p:   # lanzas 4 hilos reales (procesos)
        p.map(run_experiment, configs)