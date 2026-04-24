import pennylane as qml
import numpy as np
from PIL import Image
import time
import inicializa_params
import funciones_estado
import torch
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import read_sim_data_for_training as reader
import generador_datos
import os





# ------------------ Configuración variables iniciales ------------------

def ejecutar_autoencoder(dense_valor, dataset, n_global, n_local, n_layers_val, n_train, n_test, ansatz, mejora):
    
    denseAngle = dense_valor

    if dataset == "MNIST":

        # DATOS MNIST

        path_train = "/mnt/datastore-data-analytics/KUBIBIT/DataEncoding/mnist-png/train"
        path_test = "/mnt/datastore-data-analytics/KUBIBIT/DataEncoding/mnist-png/test"


        files = generador_datos.cargar_por_clases(path_train, n_train)
        files_test = generador_datos.cargar_por_clases(path_test, n_test)

        resize_dim = (28, 28)

    elif dataset == "SAR":
        # DATOS SAR: 

        path = "/mnt/datastore-data-analytics/KUBIBIT/DataEncoding/SAR_Dataset"

        files, files_test = generador_datos.split_dataset(
            path,
            n_train=40,
            n_test=10
        )

        resize_dim = (100, 100)
    
    # #DATOS PHASE-FIELD
    # data_raw_path = "//datastore.tekniker.es/ia/data-analytics/KUBIBIT/QML/dataset/raw_dendrites_juan"
    # idxToSave = [1500, 4000]
    # exp_id = 5
    # X, Y, _ = reader.read_simulations_nad_get_XY(None, data_raw_path, idxToSave, exp_id, sep = ";")

    tasa_de_aprendizaje = 0.05 # Tasa de aprendizaje para el optimizador

    output_block_size_height = 1   # Porque tiene 4 Z-vals → 2×2 image
    output_block_size_width = 2 # Porque tiene 4 Z-vals → 2×2 image

    block_size = 2 # Tamaño de bloque para dividir la imagen, por ejemplo, bloques de 4x4 píxeles
    
    n_blocks_h = resize_dim[0] // block_size
    n_blocks_w = resize_dim[1] // block_size


    compressed_dim = (
        n_blocks_h * output_block_size_height,
        n_blocks_w * output_block_size_width
    )


    # Dimesiones comprimida
    compressed_rows = compressed_dim[0]   
    compressed_cols = compressed_dim[1]
    
    dagger = "True"

    n_qubits_total = 3  # 2 para la información del bloque + 1 para el trash
    n_qubits_utiles = 2


    log_ratios = [] # Lista para almacenar los ratios de compresión de cada imagen procesada
    log_tamaño_original = [] # Lista para almacenar los tamaños originales de las imágenes
    log_tamaño_comprimido = [] # Lista para almacenar los tamaños comprimidos de las imágenes

    num_iteraciones_global = n_global # Número de iteraciones globales para el entrenamiento, es decir, cuántas veces se optimizan todos los bloques de la imagen
    num_iteraciones_bloque = n_local # Número de iteraciones locales para optimizar cada bloque individualmente


    optimizer_name = "Adam" # Nombre del optimizador a usar

    if denseAngle == "True":
        tecnica_de_encoding_ansatz = "DenseAngle"
    else:
        tecnica_de_encoding_ansatz = "Amplitude"

    entrenamiento = True # Si se quiere entrenar el autoencoder o solo hacer la reconstrucción con parámetros ya entrenados


    #num_layers = 2 # Número de capas para los parámetros del encoder y decoder

    # CHECKEAR EL ENTRENAMIENTO DEL ROTACIONAL EN EL DECODER
    mse_por_bloque = [] # Lista para almacenar los MSE por bloque durante la reconstrucción
    mse_por_bloque_intensity = [] # Lista para almacenar los MSE por bloque durante la reconstrucción teniendo en cuenta la intensidad de los píxeles (escalando la imagen reconstruida a su rango original de 0-255 antes de calcular el MSE)
    mse_por_imagen = [] # Lista para almacenar el MSE medio de cada imagen completa durante la reconstrucción
    mse_por_imagen_intensity = [] # Lista para almacenar el MSE medio con intensidad de píxeles de cada imagen completa durante la reconstrucción
    ssim_por_imagen = [] # Lista para almacenar el SSIM de cada imagen completa durante la reconstrucción


    start_time = time.time() # Tiempo de inicio para medir el tiempo total de ejecución

    dev = qml.device("default.qubit", wires=n_qubits_total) # Dispositivo cuántico simulado (el de por defecto)
    device = "cpu" # Dispositivo para PyTorch (CPU o GPU)

    train_mse_history = []      # MSE medio por iteración
    train_iter_history = []     # Índice global de iteración
    global_iter = 0


    compressed_img_small = np.zeros((compressed_rows, compressed_cols))
    reconstructed_img_small = np.zeros((resize_dim[0], resize_dim[1]))


    # para = 0

    idx = 0
    num_imagen = 0

    n_layers = n_layers_val
    qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits_total)
    params = inicializa_params.inic_params_angle(block_size, resize_dim, n_qubits_utiles, n_layers, ansatz)

    #Inicializar circuitos (no se usan los circuitos dev y dev_dec, pero si las funciones qnode que crean)
    autoencoder_circuit = funciones_estado.inicializar_autoencoder_amplitude(dev)
    autoencoder_circuit_dagger = funciones_estado.inicializar_autoencoder_amplitude_dagger(dev)
    circuit_enc = funciones_estado.inicializa_encoder_probs_ampl(dev)

    opt = inicializa_params.crear_optimizador(optimizer_name, params, tasa_de_aprendizaje)


    # for idx_img in range(Y.shape[0]): #PHASE-FIELD
    for f in files:
        num_imagen += 1

        # print(f"Reconstruyendo imagen comprimida {num_imagen}/{len(files)}...")

        reconstructed_blocks = []
        # img = Image.open(f).resize(resize_dim) # Redimensionar

        # img_np = X[idx_img, :, :, 0]   # canal op PHASE-FIELD
        # img_min = img_np.min()
        # img_max = img_np.max()

        # img_np = (img_np - img_min) / (img_max - img_min + 1e-8)
        # img_np = img_np * 255.0
        # img_array = torch.from_numpy(img_np).float().to(device)


        img = Image.open(f)
        # Redimensionar la imagen a las dimensiones especificadas
        img = img.resize(resize_dim)
        img_array = torch.from_numpy(np.array(img)).float().to(device)


        # Generar listas para guardar mse e iteraciones locales por imagen, de forma que luego por cada iteración global tengo una gráfica
        # Para ello quiero hacer listas de estos valores por indices, de forma que por cada iteracion global tengo una grafica donde sus valores estan guardados en las listas con los indices


        if entrenamiento == True:
            # print("\n=== ENTRENAMIENTO AUTOENCODER ===")

            for epoch in range(num_iteraciones_global):
                # print(f"\nEpoch {epoch+1}/{num_iteraciones_global}")

                mse_epoch = []
                iter_epoch = []

                bloque_num = 0
                for i in range(0, resize_dim[0], block_size):
                    for j in range(0, resize_dim[1], block_size):
                        bloque_num+=1
                        # print(f"\nBloque {bloque_num}/{(resize_dim[0]*resize_dim[1])/(block_size*block_size)}")

                        state_sin_norm, block_norm, block_sum, norm = funciones_estado.imagen_flatten(
                            img_array, i, j, block_size, device
                        )

                        if norm > 1e-12:
                                state = state_sin_norm / norm
                        else:
                                state = state_sin_norm  # Si la norma es muy pequeña, usar el estado sin normalizar para evitar división por cero
                            
                        # if(block_sum.item() == 0.0):
                        #     # print("Bloque de solo ceros, saltando entrenamiento local del bloque.")
                        #     continue

                        # grafiquito_mse = []
                        for iter in range(num_iteraciones_bloque):

                            # ---- ENTRENAMIENTO ----
                            params[(i, j)], loss, mse = funciones_estado.optimizar_autoencoder_bloque(
                                opt, params[(i, j)], state, autoencoder_circuit, autoencoder_circuit_dagger, dagger, denseAngle, ansatz, mejora
                            )


                            # print(f" Iteración {iter+1}/{num_iteraciones_bloque} - MSE: {mse:.6f}")

                            # grafiquito_mse.append(mse)

                            if iter == 3 and mse < 1e-6:
                                # print("MSE muy bajo, saliendo del entrenamiento local del bloque.")
                                break

                        # #Grafico rapido y simple
                        # para = para + 1
                        # if para <= 3:
                        #     plt.figure()
                        #     plt.plot(grafiquito_mse)
                        #     plt.xlabel("Iteración")
                        #     plt.ylabel("MSE")
                        #     plt.title("Evolución del MSE durante el entrenamiento")
                        #     plt.show()

                        global_iter += 1
                        mse_epoch.append(mse)
                        iter_epoch.append(global_iter)

                train_mse_history.append(mse_epoch)
                train_iter_history.append(iter_epoch)
        # funciones_estado.graficar_MSE(train_iter_history, train_mse_history)
        train_mse_history = []
        train_iter_history = []



    for f in files_test:
        base_dir = "/mnt/datastore-data-analytics/KUBIBIT/DataEncoding"
        numero = os.path.basename(os.path.dirname(f))

        carpeta_intermedia = (
            f"{tecnica_de_encoding_ansatz}_"
            f"glob{num_iteraciones_global}_"
            f"loc{num_iteraciones_bloque}_"
            f"layers{n_layers}_"
            f"train{n_train * 10}_"
            f"test{n_test * 10}"
        )

        mejora_str = "Con Mejora" if mejora else "Sin Mejora"

        output_dir = os.path.join(
            base_dir,
            "Resultados",
            dataset,
            ansatz,
            mejora_str,
            carpeta_intermedia,
            numero
        )


        os.makedirs(output_dir, exist_ok=True)
        nombre = os.path.basename(f)

        reconstructed_save_path = os.path.join(output_dir, f"reconstructed_{nombre}")
        compressed_save_path = os.path.join(output_dir, f"compressed_{nombre}")
        img_save_path = os.path.join(output_dir, f"original_{nombre}")
        combined_save_path = os.path.join(output_dir, f"comparacion_{nombre}")


        img = Image.open(f)
        # Redimensionar la imagen a las dimensiones especificadas
        img = img.resize(resize_dim)
        img_array = torch.from_numpy(np.array(img)).float().to(device)
        # print("\n=== RECONSTRUCCIÓN ===")


        for i in range(0, resize_dim[0], block_size): # Iterar sobre la imagen en pasos del tamaño del bloque (filas)
            for j in range(0, resize_dim[1], block_size): # Iterar sobre la imagen en pasos del tamaño del bloque (columnas)
                state_sin_norm, block_norm, block_sum, norm = funciones_estado.imagen_flatten(img_array, i, j, block_size, device)

                if norm > 1e-12:
                        state = state_sin_norm / norm
                else:
                        state = state_sin_norm  # Si la norma es muy pequeña, usar el estado sin normalizar para evitar división por cero
                # print("\nEstado inicial a utilizar en bloque :", state)

                params_enc, params_dec = params[(i, j)]

                vals_enc = circuit_enc(state , params_enc, denseAngle, ansatz, mejora)
                # print("Valores de probabilidad obtenidos por el encoder : ",vals_enc)
                if denseAngle == "True":
                    vals_enc = (1 - torch.stack(vals_enc)) / 2
                else:
                    vals_enc = torch.sqrt(vals_enc)
                # vals_enc = torch.sqrt(vals_enc) * norm
                # print("Raíz de los valores de probabilidad obtenidos por el encoder : ",vals_enc)
                
                # Índices destino en la imagen comprimida
                row_idx = (i // block_size) * output_block_size_height
                col_idx = (j // block_size) * output_block_size_width

                # print("Z valores obtenidos por el encoder : ",z_vals)
                # print("Block sum (intensidad del bloque original) : ", block_sum)

                output = funciones_estado.escalar_generar_imagen_mediciones_encoder(vals_enc, output_block_size_height, output_block_size_width, norm, "False")
                # print("Valores de la imagen comprimida obtenidos por el encoder : ", output)

                # print("Output del encoder : ", output)

                compressed_img_small[row_idx:row_idx+output_block_size_height,
                                col_idx:col_idx+output_block_size_width] = output
                            
                # Decoder 


                if dagger == "True":
                    recon, _ = autoencoder_circuit_dagger(state, params_enc, params_dec, denseAngle, ansatz, mejora)
                else:
                    recon, _ = autoencoder_circuit(state, params_enc, params_dec, denseAngle, ansatz, mejora)

                # print("Valores de probabilidad obtenidos por el decoder : ", recon)
                if denseAngle == "True":
                    recon = (1 - torch.stack(recon)) / 2
                else:
                    recon = torch.sqrt(recon)
                # recon = torch.sqrt(recon) * norm  # Escalado a la intensidad original del bloque

                # print("Raíz de los valores de probabilidad obtenidos por el decoder : ", recon)
                output_dec = funciones_estado.escalar_generar_imagen_mediciones_decoder(recon, block_size, norm, "False")
                # print("Valores de la imagen reconstruida obtenidos por el decoder : ", output_dec)

                reconstructed_img_small[i:(i+block_size), 
                                        j:(j+block_size)] = output_dec

                if(block_sum.item() == 0.0):
                    mse = 0.0
                    mse_intensity = 0.0

                else:
                    #MSE DEL ENTRENAMIENTO DIRECTAMENTE
                    recon = recon.flatten()
                    mse = torch.mean((state.flatten() - recon)**2)

                    #MSE TRAS MULTIPLICARLE EL VALOR DEL FACTOR DE NORMALIZACIÓN, OSEA OBTENER EL ESTADO SIN NORMALIZAR
                    output01 = output_dec / 255.0
                    ouput01_flatten = output01.flatten()
                    mse_intensity = torch.mean((state.flatten() * norm - ouput01_flatten)**2)
                # print(f"\nMSE: {mse:.6f}")


                mse_por_bloque.append(mse)
                mse_por_bloque_intensity.append(mse_intensity)
        
                idx += 1

        # Guardar las imágenes resultantes

        Image.fromarray(compressed_img_small.astype(np.uint8)).save(compressed_save_path)
        Image.fromarray(img_array.detach().cpu().numpy().astype(np.uint8)).save(img_save_path)    
        Image.fromarray(reconstructed_img_small.astype(np.uint8)).save(reconstructed_save_path)

        # funciones_estado.graficar(
        #     img_array, resize_dim,
        #     compressed_img_small, compressed_dim,
        #     reconstructed_img_small,
        #     show_values=True
        # )

        funciones_estado.graficar(img_array, resize_dim, compressed_img_small, compressed_dim, reconstructed_img_small, combined_save_path)
        # funciones_estado.prueba(img_array, resize_dim, compressed_img_small, compressed_dim, reconstructed_img_small, block_size)


        # inicial_original_disk_size = funciones_estado.obtener_tamaño(f)
        original_disk_size = funciones_estado.obtener_tamaño(img_save_path)
        compressed_disk_size = funciones_estado.obtener_tamaño(compressed_save_path)
        disk_ratio = original_disk_size / compressed_disk_size if compressed_disk_size > 0 else float('inf')

        # Guardar ratio de compresión de cada imagen
        log_tamaño_original.append(original_disk_size)
        log_tamaño_comprimido.append(compressed_disk_size)
        log_ratios.append(disk_ratio)
        # print(f"Imagen {num_imagen}: Tamano original = {original_disk_size:.6f} MB, Tamaño comprimido = {compressed_disk_size:.6f} MB, Ratio de compresion = {disk_ratio:.2f}x")
            
        # ------------------ Métricas ------------------

        #Media de MSE global
        mse_nonzero = [x for x in mse_por_bloque if x != 0.0] #  Filtrar los MSE que son exactamente 0.0 (bloques de solo ceros) para no afectar la media
        media_MSE = sum(mse_nonzero) / len(mse_nonzero) if mse_nonzero else 0.0


        #Media de MSE con intensidad de píxeles
        mse_intensity_nonzero = [x for x in mse_por_bloque_intensity if x != 0.0]
        average_mse_intensity = sum(mse_intensity_nonzero) / len(mse_intensity_nonzero) if mse_intensity_nonzero else 1.0

        original = img_array.detach().cpu().numpy().astype(np.uint8)
        reconstructed = reconstructed_img_small.astype(np.uint8)
        ssim_val = ssim(original, reconstructed, data_range=255)

        mse_por_imagen.append(media_MSE) # Guardar el MSE medio de la imagen completa en la lista correspondiente para luego calcular la media global de MSE por imagen
        mse_por_imagen_intensity.append(average_mse_intensity) # Guardar el MSE medio con intensidad de píxeles de la imagen completa en la lista correspondiente para luego calcular la media global de MSE por imagen
        ssim_por_imagen.append(ssim_val) # Guardar el SSIM de la imagen completa en la lista correspondiente para luego calcular la media global de SSIM por imagen

    #obtner media de MSE por todas las imagenes
    media_final_MSE = sum(mse_por_imagen) / len(mse_por_imagen) if mse_por_imagen else 0.0
    media_final_MSE_intensity = sum(mse_por_imagen_intensity) / len(mse_por_imagen_intensity) if mse_por_imagen_intensity else 0.0
    media_final_SSIM = sum(ssim_por_imagen) / len(ssim_por_imagen ) if ssim_por_imagen else 0.0



    # Media de ratios de compresión
    average_disk_ratio = np.mean(log_ratios) if log_ratios else 0
    average_tamaño_original = np.mean(log_tamaño_original) if log_tamaño_original else 0
    average_tamaño_comprimido = np.mean(log_tamaño_comprimido) if log_tamaño_comprimido else 0



    end_time = time.time() # Tiempo de fin para medir el tiempo total de ejecución
    tiempo_total = end_time - start_time
    # print(f"Tiempo total: {tiempo_total:.2f} segundos")

    ruta_log = os.path.join(
        base_dir,
        "Resultados",
        dataset,
        ansatz,
        mejora_str,
        f"{carpeta_intermedia}.txt"
    )

    # Crear carpetas si no existen
    os.makedirs(os.path.dirname(ruta_log), exist_ok=True)

    funciones_estado.guardar_log(
        ruta_log, 
        resize_dim, 
        compressed_dim, 
        block_size, 
        2, 
        tecnica_de_encoding_ansatz, 
        dataset, 
        n_train * 10, 
        tiempo_total, 
        average_disk_ratio, 
        average_tamaño_original, 
        average_tamaño_comprimido, 
        num_iteraciones_global, 
        num_iteraciones_bloque, 
        tasa_de_aprendizaje, 
        n_layers,
        n_test * 10,
        ansatz
    )




    with open (ruta_log, "a") as f:
        f.write(f"\nMSE medio global de la imagen: {media_final_MSE:.6f}\n")
        f.write(f"SSIM medio de las imagenes: {media_final_SSIM:.6f}\n")
        f.write(f"MSE medio con intensidad de pixeles: {media_final_MSE_intensity:.6f}\n")

