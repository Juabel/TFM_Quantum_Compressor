import pennylane as qml
import numpy as np
from PIL import Image
import time
import inicializa_params
import funciones_estado
import torch
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import generador_datos
import os



# ------------------ Configuración variables iniciales ------------------


def ejecutar_autoencoder(basis_valor, dataset, n_global, n_local, n_layers_val, n_train, n_test, ansatz, mejora):
    start_time = time.time() # Tiempo de inicio para medir el tiempo total de ejecución

    start_time_preproceso = time.time() # Tiempo de inicio para medir el tiempo del preprocesamiento de datos

    basis = basis_valor
    #DATOS SATELITALES

    if dataset == "MNIST":

        # DATOS MNIST
       
        path_train = r"\\datastore.tekniker.es\ia\data-analytics\KUBIBIT\DataEncoding\mnist-png\train"
        path_test  = r"\\datastore.tekniker.es\ia\data-analytics\KUBIBIT\DataEncoding\mnist-png\test"

        files = generador_datos.cargar_por_clases(path_train, n_train)
        files_test = generador_datos.cargar_por_clases(path_test, n_test)

        resize_dim = (28, 28)
        mse_por_clase = {str(i): [] for i in range(10)}


    elif dataset == "SAR":
        # DATOS SAR: 

        path = r"\\datastore.tekniker.es\ia\data-analytics\KUBIBIT\DataEncoding\SAR_Dataset"

        files, files_test = generador_datos.split_dataset(
            path,
            n_train=40,
            n_test=10
        )

        resize_dim = (100, 100)
        mse_por_clase = {str(i): [] for i in range(2)}
        


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

    dagger = "True" # Si se quiere usar el autoencoder dagger para la parte del decoder, es decir, usar el mismo circuito pero con los parámetros en orden inverso y con los ángulos negados. Esto se hace para ver si el entrenamiento del encoder es suficiente para que el decoder aprenda a reconstruir la imagen sin necesidad de entrenar específicamente el decoder, lo cual sería una ventaja importante en términos de eficiencia de entrenamiento.
   
    n_qubits_total = 6 # 4 + 2 ancilla 
    n_qubits_utiles = 4


    log_ratios = [] # Lista para almacenar los ratios de compresión de cada imagen procesada
    log_tamaño_original = [] # Lista para almacenar los tamaños originales de las imágenes
    log_tamaño_comprimido = [] # Lista para almacenar los tamaños comprimidos de las imágenes

    num_iteraciones_global = n_global # Número de iteraciones globales para el entrenamiento, es decir, cuántas veces se optimizan todos los bloques de la imagen
    num_iteraciones_bloque = n_local # Número de iteraciones locales para optimizar cada bloque individualmente


    optimizer_name = "Adam" # Nombre del optimizador a usar
    if basis == "True":
        tecnica_de_encoding_ansatz = "Basis"
    else:
        tecnica_de_encoding_ansatz = "Angle"

    entrenamiento = True # Si se quiere entrenar el autoencoder o solo hacer la reconstrucción con parámetros ya entrenados


    # CHECKEAR EL ENTRENAMIENTO DEL ROTACIONAL EN EL DECODER
    mse_por_imagen = [] # Lista para almacenar el MSE medio de cada imagen completa durante la reconstrucción
    mse_por_imagen_intensity = [] # Lista para almacenar el MSE medio con intensidad de píxeles de cada imagen completa durante la reconstrucción
    ssim_por_imagen = [] # Lista para almacenar el SSIM de cada imagen completa durante la reconstrucción


    start_time = time.time() # Tiempo de inicio para medir el tiempo total de ejecución

    dev = qml.device("default.qubit", wires=n_qubits_total) # Dispositivo cuántico simulado (el de por defecto)
    device = "cpu" # Dispositivo para PyTorch (CPU o GPU)

    # --- Métricas de entrenamiento ---
    train_mse_history = []      # MSE medio por iteración
    train_mse_imagenes = []      # Lista para almacenar el historial de MSE por iteración global de cada imagen

    compressed_img_small = np.zeros((compressed_rows, compressed_cols))
    reconstructed_img_small = np.zeros((resize_dim[0], resize_dim[1]))


    idx = 0
    num_imagen = 0


    n_layers = n_layers_val
    qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits_total)

    params = inicializa_params.inic_params_angle(block_size, resize_dim, n_qubits_utiles, n_layers, ansatz)

    #Inicializar circuitos (no se usan los circuitos dev y dev_dec, pero si las funciones qnode que crean)
    autoencoder_recon = funciones_estado.inicializar_autoencoder(dev)
    autoencoder_recon_dagger = funciones_estado.inicializar_autoencoder_dagger(dev)
    circuit_encoder_probs = funciones_estado.inicializa_encoder_probs(dev)


    opt = inicializa_params.crear_optimizador(optimizer_name, params, tasa_de_aprendizaje)

    base_dir = r"\\datastore.tekniker.es\ia\data-analytics\KUBIBIT\DataEncoding"
    carpeta_intermedia = (
    f"{tecnica_de_encoding_ansatz}_"
    f"glob{num_iteraciones_global}_"
    f"loc{num_iteraciones_bloque}_"
    f"layers{n_layers}_"
    f"train{n_train * 10}_"
    f"test{n_test * 10}"
)
    mejora_str = "Con Mejora" if mejora else "Sin Mejora"


    end_time_preproceso = time.time() # Tiempo de fin para medir el tiempo del preprocesamiento de datos
    tiempo_preproceso = end_time_preproceso - start_time_preproceso


    start_time_entrenamiento = time.time()

    lista_mse_final_bloque = []
    for f in files:

        numero = os.path.basename(os.path.dirname(f))

        output_dir = os.path.join(
            base_dir,
            "Resultados",
            dataset,
            ansatz,
            mejora_str,
            carpeta_intermedia,
            numero
        )
        # print(f"Reconstruyendo imagen comprimida {num_imagen}/{len(files)}...")

        num_imagen += 1
        # print(f"Reconstruyendo imagen comprimida {num_imagen}/{len(files)}...")
        reconstructed_blocks = []
        # img = Image.open(f).resize(resize_dim) # Redimensionar
        img = funciones_estado.open_image_safe(f)
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

                        if basis == "True":
                            state = (state_sin_norm > 0.5).float()
                        else:
                            if norm > 1e-12:
                                state = state_sin_norm / norm
                            else:
                                state = state_sin_norm  # Si la norma es muy pequeña, usar el estado sin normalizar para evitar división por cero

                        # if(block_sum.item() == 0.0):
                        #     # print("Bloque de solo ceros, saltando entrenamiento local del bloque.")
                        #     continue
                        grafiquito_mse = []
                        for iter in range(num_iteraciones_bloque):

                            params_enc, params_dec = params[(i, j)]

                            # qml.drawer.use_style("black_white")

                            # fig, ax = qml.draw_mpl(encoder_trash)(
                            #     state,
                            #     params_enc,
                            # )
                            # plt.show()

                            # ---- ENTRENAMIENTO ----
                            params_enc, params_dec, loss, mse = funciones_estado.optimizar_autoencoder_bloque_angle(
                                opt, params_enc, params_dec, state, autoencoder_recon, autoencoder_recon_dagger, dagger, n_qubits_utiles, n_qubits_total, basis, ansatz, mejora
                            )

                            params[(i, j)] = (params_enc, params_dec)

                            # print(f" Iteración {iter+1}/{num_iteraciones_bloque} - MSE: {mse:.6f}")

                            # Dejamos el entrenamiento cuando el MSE es menor que un umbral, para evitar que luego se equivoque y empiece a subir el MSE por sobreentrenamiento
                            # if iter > 0 and mse < 0.07:
                            #     print("MSE menor que 0.07, saliendo del entrenamiento local del bloque.")
                            #     break
                            # grafiquito_mse.append(mse)

                            if iter == 3 and mse < 1e-6:
                                # print("MSE muy bajo, saliendo del entrenamiento local del bloque.")
                                break

                        #Grafico rapido y simple
                        # para = para + 1
                        # if para <= 3:
                        #     plt.figure()
                        #     plt.plot(grafiquito_mse)
                        #     plt.xlabel("Iteración")
                        #     plt.ylabel("MSE")
                        #     plt.title("Evolución del MSE durante el entrenamiento")
                        #     plt.show()

                        mse_epoch.append(mse)
                if epoch == num_iteraciones_global - 1:
                    train_mse_history.append(mse_epoch)            

                mse_epoch = []
        # funciones_estado.graficar_MSE(train_iter_history, train_mse_history)
        train_mse_imagenes.append(train_mse_history) # Guardar el historial de MSE por iteración global de cada imagen en la lista correspondiente para luego analizar el comportamiento del entrenamiento a lo largo de las iteraciones globales y comparar entre imágenes
        train_mse_history = []
    save_dir = os.path.dirname(output_dir)
    os.makedirs(save_dir, exist_ok=True)
    funciones_estado.comportamiento_entrenamiento_global(train_mse_imagenes, save_dir)

    latentes = []
    labels = []
    heatmap = np.zeros(resize_dim)
    contador = np.zeros(resize_dim)

    end_time_entrenamiento = time.time()
    tiempo_entrenamiento = end_time_entrenamiento - start_time_entrenamiento

    start_time_reconstruccion = time.time()


    for f in files_test:
        mse_por_bloque = [] # Lista para almacenar los MSE por bloque durante la reconstrucción
        mse_por_bloque_intensity = [] # Lista para almacenar los MSE por bloque durante la reconstrucción teniendo en cuenta la intensidad de los píxeles (escalando la imagen reconstruida a su rango original de 0-255 antes de calcular el MSE)
  
        numero = os.path.basename(os.path.dirname(f))

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


        img = funciones_estado.open_image_safe(f)
        # Redimensionar la imagen a las dimensiones especificadas
        img = img.resize(resize_dim)
        img_array = torch.from_numpy(np.array(img)).float().to(device)
        # print("\n=== RECONSTRUCCIÓN ===")

        z_imagen = []

        for i in range(0, resize_dim[0], block_size): # Iterar sobre la imagen en pasos del tamaño del bloque (filas)
            for j in range(0, resize_dim[1], block_size): # Iterar sobre la imagen en pasos del tamaño del bloque (columnas)
                state_sin_norm, block_norm, block_sum, norm = funciones_estado.imagen_flatten(img_array, i, j, block_size, device)

                if basis == "True":
                    state = (state_sin_norm > 0.5).float()
                else:
                    if norm > 1e-12:
                        state = state_sin_norm / norm
                    else:
                        state = state_sin_norm  # Si la norma es muy pequeña, usar el estado sin normalizar para evitar división por cero


                params_enc, params_dec = params[(i, j)]


                z_vals_encoder = circuit_encoder_probs(state, params_enc, n_qubits_utiles, n_qubits_total, basis, ansatz, mejora)


                # compressed_pixels = funciones_estado.coarse_grain_probs(z_vals_encoder, n_pixels_out)
                z_vals_encoder = torch.stack(z_vals_encoder)

                            
                # Índices destino en la imagen comprimida
                row_idx = (i // block_size) * output_block_size_height
                col_idx = (j // block_size) * output_block_size_width

                z_vals_encoder = (1 - z_vals_encoder) / 2

                z_vals = torch.stack(z_vals_encoder) if isinstance(z_vals_encoder, list) else z_vals_encoder
                z_imagen.extend(z_vals.detach().cpu().numpy().flatten())                

                output = funciones_estado.escalar_generar_imagen_mediciones_encoder(z_vals_encoder, output_block_size_height, output_block_size_width, norm, basis)



                compressed_img_small[row_idx:row_idx+output_block_size_height,
                                col_idx:col_idx+output_block_size_width] = output



                # Decoder 
                if dagger == "True":
                    z_vals_decoder, _ = autoencoder_recon_dagger(state, params_enc, params_dec, n_qubits_utiles, n_qubits_total, basis, ansatz, mejora)
                else:
                    z_vals_decoder, _ = autoencoder_recon(state, params_enc, params_dec, n_qubits_utiles, n_qubits_total, basis, ansatz, mejora)
                # z_vals_decoder = circuit_decoder_probs(z_vals_encoder, params_dec)
                # z_vals_decoder = funciones_estado.coarse_grain_probs(z_vals_decoder, n_qubits)

                z_vals_decoder = torch.stack(z_vals_decoder)

                # print("Z valores obtenidos por el decoder : ", z_vals_decoder)

                z_vals_decoder = (1 - z_vals_decoder) / 2
                output_dec = funciones_estado.escalar_generar_imagen_mediciones_decoder(z_vals_decoder, block_size, norm, basis)

                # print("Output del decoder : ", output_dec)

                reconstructed_img_small[i:(i+block_size), 
                                        j:(j+block_size)] = output_dec
                
                if(block_sum.item() == 0.0):
                    mse = 0.0
                    mse_intensity = 0.0
                else:
                    #MSE SIN INTENSDIAD PIXELES
                    z_vals_decoder = z_vals_decoder.flatten()

                    # print("Estado original a utilizar en bloque : ", state)
                    # print("Output del decoder a utilizar en bloque : ", ouput01_flatten)
                    # print("Output del decoder a utilizar en bloque : ", prueba_flatten)
                    mse = torch.mean((state.flatten() - z_vals_decoder)**2).item()



                    #MSE CON INTENSIDAD PIXELES
                    output01 = output_dec / 255.0
                    ouput01_flatten = output01.flatten()
                    if basis == "True":
                        mse_intensity = torch.mean((state.flatten() - ouput01_flatten)**2).item()
                        error = np.abs(state.flatten() - ouput01_flatten)

                    else:
                        mse_intensity = torch.mean((state.flatten() * norm - ouput01_flatten)**2).item()
                        error = np.abs(state.flatten() * norm - ouput01_flatten)
                    
                    error = error.detach().cpu().numpy()
                    error_block = error.reshape(block_size, block_size)

                    heatmap[i:i+block_size, j:j+block_size] += error_block
                    contador[i:i+block_size, j:j+block_size] += 1

                # print(f"\nMSE: {mse:.6f}")

                # print("\nMeto a la lista el mse ", mse)
                mse_por_bloque.append(mse)
                mse_por_bloque_intensity.append(mse_intensity)

                idx += 1

        # Guardar las imágenes resultantes
        latentes.append(z_imagen)
        labels.append(int(numero))
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


        inicial_original_disk_size = funciones_estado.obtener_tamaño(f)
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

        mse_nonzero = [x for x in mse_por_bloque if x != 0.0]
        media_MSE = sum(mse_nonzero) / len(mse_nonzero) if mse_nonzero else 1.0


        #Media de MSE con intensidad de píxeles
        mse_intensity_nonzero = [x for x in mse_por_bloque_intensity if x != 0.0]
        average_mse_intensity = sum(mse_intensity_nonzero) / len(mse_intensity_nonzero) if mse_intensity_nonzero else 1.0

        original = img_array.detach().cpu().numpy().astype(np.uint8)
        reconstructed = reconstructed_img_small.astype(np.uint8)
        ssim_val = ssim(original, reconstructed, data_range=255)

        mse_por_imagen.append(media_MSE) # Guardar el MSE medio de la imagen completa en la lista correspondiente para luego calcular la media global de MSE por imagen
        mse_por_imagen_intensity.append(average_mse_intensity) # Guardar el MSE medio con intensidad de píxeles de la imagen completa en la lista correspondiente para luego calcular la media global de MSE por imagen
        ssim_por_imagen.append(ssim_val) # Guardar el SSIM de la imagen completa en la lista correspondiente para luego calcular la media global de SSIM por imagen

        mse_por_clase[numero].append(media_MSE)

    end_time_reconstruccion = time.time()
    tiempo_reconstruccion = end_time_reconstruccion - start_time_reconstruccion

    start_time_analisis = time.time()




    funciones_estado.histograma_MSE_por_clase(mse_por_clase, save_dir)

    error_medio = np.divide(
        heatmap,
        contador,
        out=np.zeros_like(heatmap),
        where=contador != 0
    )
    funciones_estado.mapa_de_calor(error_medio, save_dir)


    funciones_estado.graficar_TSNE(latentes, labels, save_dir)

    #obtner media de MSE por todas las imagenes
    media_final_MSE = sum(mse_por_imagen) / len(mse_por_imagen) if mse_por_imagen else 0.0
    media_final_MSE_intensity = sum(mse_por_imagen_intensity) / len(mse_por_imagen_intensity) if mse_por_imagen_intensity else 0.0
    media_final_SSIM = sum(ssim_por_imagen) / len(ssim_por_imagen ) if ssim_por_imagen else 0.0


    # Media de ratios de compresión
    average_disk_ratio = np.mean(log_ratios) if log_ratios else 0
    average_tamaño_original = np.mean(log_tamaño_original) if log_tamaño_original else 0
    average_tamaño_comprimido = np.mean(log_tamaño_comprimido) if log_tamaño_comprimido else 0


    ruta_log = os.path.join(
        base_dir,
        "Resultados",
        dataset,
        ansatz,
        mejora_str,
        carpeta_intermedia,
        f"{carpeta_intermedia}.txt"
    )
    
    ruta_tiempos = os.path.join(
        base_dir,
        "Resultados",
        dataset,
        ansatz,
        mejora_str,
        carpeta_intermedia,
        f"Tiempos.txt"
    )

    # Crear carpetas si no existen
    os.makedirs(os.path.dirname(ruta_log), exist_ok=True)
    end_time = time.time() # Tiempo de fin para medir el tiempo total de ejecución

    tiempo_total = end_time - start_time

    end_time_analisis = time.time()
    tiempo_analisis = end_time_analisis - start_time_analisis

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
        ansatz, 
        media_final_MSE,
        media_final_SSIM,
        media_final_MSE_intensity
    )

    funciones_estado.guardar_tiempos(ruta_tiempos, 
        tiempo_total, 
        tiempo_analisis, 
        tiempo_preproceso, 
        tiempo_entrenamiento, 
        tiempo_reconstruccion
    )

