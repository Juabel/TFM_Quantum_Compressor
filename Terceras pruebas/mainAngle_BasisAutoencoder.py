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
from torch.utils.data import DataLoader 
import random
from sklearn.model_selection import train_test_split
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau



# ------------------ Configuración variables iniciales ------------------


def ejecutar_autoencoder(basis_valor, dataset, n_epochs, batch_size, n_layers_val, n_train, n_test, ansatz, mejora, k, dev_size, lr):
    start_time = time.time() # Tiempo de inicio para medir el tiempo total de ejecución

    start_time_preproceso = time.time() # Tiempo de inicio para medir el tiempo del preprocesamiento de datos

    basis = basis_valor

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
            n_train,
            n_test
        )

        resize_dim = (100, 100)
        mse_por_clase = {str(i): [] for i in range(2)}

        
    files_train, files_dev = train_test_split(files, test_size=dev_size , random_state=42)


    train_dataset = generador_datos.QuantumImageDataset(files_train, resize_dim)
    dev_dataset = generador_datos.QuantumImageDataset(
        files_dev,
        resize_dim
    )

    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False
    )


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

   
    n_qubits_total = 6 # 4 + 2 ancilla 
    n_qubits_utiles = 4


    optimizer_name = "Adam" # Nombre del optimizador a usar
    if basis == "True":
        tecnica_de_encoding_ansatz = "Basis"
    else:
        tecnica_de_encoding_ansatz = "Angle"


    # CHECKEAR EL ENTRENAMIENTO DEL ROTACIONAL EN EL DECODER
    mse_por_imagen = [] # Lista para almacenar el MSE medio de cada imagen completa durante la reconstrucción
    ssim_por_imagen = [] # Lista para almacenar el SSIM de cada imagen completa durante la reconstrucción

    train_loss_history = []
    dev_loss_history = []

    batch_loss_history = []
    grad_history = []
    block_loss_history = []
    block_loss_history_all_epochs = []

    dev = qml.device("default.qubit", wires=n_qubits_total) # Dispositivo cuántico simulado (el de por defecto)
    device = "cpu" # Dispositivo para PyTorch (CPU o GPU)

    compressed_img_small = np.zeros((compressed_rows, compressed_cols))
    reconstructed_img_small = np.zeros((resize_dim[0], resize_dim[1]))


    n_layers = n_layers_val
    qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits_total)

    params_enc = inicializa_params.inic_params_angle(n_qubits_utiles, n_layers, ansatz)

    #Inicializar circuitos (no se usan los circuitos dev y dev_dec, pero si las funciones qnode que crean)
    autoencoder_recon_dagger = funciones_estado.inicializar_autoencoder_dagger(dev)
    circuit_encoder_probs = funciones_estado.inicializa_encoder_probs(dev)

    tasa_de_aprendizaje = lr # Tasa de aprendizaje para el optimizador

    opt = inicializa_params.crear_optimizador(optimizer_name, params_enc, tasa_de_aprendizaje)
    scheduler = ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=1,
        threshold=1e-5,
        threshold_mode="abs",
        min_lr=1e-8
    )

    num_epochs = n_epochs

    if dataset == "MNIST":
        train_nombre = n_train * 10
        test_nombre = n_test * 10
    else:
        train_nombre = n_train * 2
        test_nombre = n_test * 2


    best_dev_loss = float("inf")
    best_params = None

    base_dir = r"\\datastore.tekniker.es\ia\data-analytics\KUBIBIT\DataEncoding"
    carpeta_intermedia = (
    f"{tecnica_de_encoding_ansatz}_"
    f"epochs{num_epochs}_"
    f"batchsize{batch_size}_"
    f"layers{n_layers}_"
    f"train{train_nombre}_"
    f"test{test_nombre}_"
    f"bloques{k}_"
    f"lr{tasa_de_aprendizaje}"
)
    mejora_str = "Con Mejora" if mejora else "Sin Mejora"


    end_time_preproceso = time.time() # Tiempo de fin para medir el tiempo del preprocesamiento de datos
    tiempo_preproceso = end_time_preproceso - start_time_preproceso


    start_time_entrenamiento = time.time()

    for epoch in range(num_epochs):

        epoch_loss = 0.0

        print(f"\n================ EPOCH {epoch+1}/{num_epochs} ================")

        for batch_idx, (images, paths) in enumerate(train_loader):
            images = images.to(device)
            opt.zero_grad()
            batch_total_loss = torch.tensor(0.0, device=device)

            total_valid_images = 0

            for img_idx in range(images.shape[0]):
                img = images[img_idx]
                total_img_loss = torch.tensor(0.0, device=device)
                
                K_bloques = k

                valid_coords = funciones_estado.get_valid_blocks(img, block_size) 
                sampled_blocks = random.sample(valid_coords,min(K_bloques, len(valid_coords)))

                valid_blocks = 0

                for (i, j) in sampled_blocks:
                    block = img[:, i:i+block_size, j:j+block_size]

                    state_sin_norm, block_sum, norm = funciones_estado.imagen_flatten_batch(block, device)

                    if norm > 1e-12:
                        state = state_sin_norm / norm
                    else:
                        state = state_sin_norm  # Normalizar el estado si la norma es suficientemente grande, de lo contrario usar el estado sin normalizar para evitar división por cero

                    if basis == "True" and norm > 1e-12:
                        state = (state > 0.5).int()
            

                    loss, mse = funciones_estado.optimizar_autoencoder_bloque_angle(
                        params_enc, state, autoencoder_recon_dagger, n_qubits_utiles, n_qubits_total, basis, ansatz, mejora
                    )

                    total_img_loss += loss
                    valid_blocks += 1
                    block_loss_history.append(loss.item())

                if valid_blocks == 0:
                    continue

                total_img_loss = total_img_loss / valid_blocks

                batch_total_loss += total_img_loss

                total_valid_images += 1

            if total_valid_images == 0:
                continue

            batch_total_loss = batch_total_loss / total_valid_images
            
            batch_total_loss.backward()

            grad_norm = params_enc.grad.norm().item()
            grad_history.append(grad_norm)
            
            opt.step()

            batch_loss = batch_total_loss.item()
            batch_loss_history.append(batch_loss)

            epoch_loss += batch_loss

            print(f"Batch {batch_idx+1}/{len(train_loader)} - Loss: {batch_loss:.6f}")

        block_loss_history_all_epochs.append(block_loss_history)
        block_loss_history = []

        avg_epoch_loss = epoch_loss / len(train_loader)
        
        train_loss_history.append(avg_epoch_loss)

        print(f"\nEpoch {epoch+1} finalizada")
        print(f"Loss media epoch: {avg_epoch_loss:.6f}")



        dev_epoch_loss = 0.0

        with torch.no_grad():

            for batch_idx, (images, paths) in enumerate(dev_loader):

                images = images.to(device)

                batch_total_loss = 0.0

                total_valid_images = 0

                for img_idx in range(images.shape[0]):

                    img = images[img_idx]

                    total_img_loss = 0.0

                    valid_coords = funciones_estado.get_valid_blocks(
                        img,
                        block_size
                    )

                    # IMPORTANTE:
                    # En validation usamos TODOS los bloques
                    sampled_blocks = valid_coords

                    valid_blocks = 0

                    for (i, j) in sampled_blocks:

                        block = img[:, i:i+block_size, j:j+block_size]

                        state_sin_norm, block_sum, norm = (
                            funciones_estado.imagen_flatten_batch(
                                block,
                                device
                            )
                        )

                        if norm > 1e-12:
                            state = state_sin_norm / norm
                        else:
                            state = state_sin_norm

                        if basis == "True" and norm > 1e-12:
                            state = (state > 0.5).int()
                        
                        loss, mse = funciones_estado.optimizar_autoencoder_bloque_angle(
                            params_enc, state, autoencoder_recon_dagger, n_qubits_utiles, n_qubits_total, basis, ansatz, mejora
                        )

                        total_img_loss += loss.item()

                        valid_blocks += 1

                    if valid_blocks == 0:
                        continue

                    total_img_loss = total_img_loss / valid_blocks

                    batch_total_loss += total_img_loss

                    total_valid_images += 1

                if total_valid_images == 0:
                    continue

                batch_total_loss = (
                    batch_total_loss / total_valid_images
                )

                dev_epoch_loss += batch_total_loss

        avg_dev_loss = dev_epoch_loss / len(dev_loader)

        dev_loss_history.append(avg_dev_loss)

        scheduler.step(avg_dev_loss)
        current_lr = opt.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1} | "
            f"Train {avg_epoch_loss:.6f} | "
            f"Dev {avg_dev_loss:.6f} | "
            f"LR {opt.param_groups[0]['lr']:.6f}"
        )


        # =====================================================
        # GUARDAR MEJOR MODELO
        # =====================================================

        if avg_dev_loss < best_dev_loss:

            best_dev_loss = avg_dev_loss

            best_params = copy.deepcopy(params_enc)

            print("Nuevo mejor modelo guardado")

    # =====================================================
    # CARGAR MEJOR MODELO SEGÚN DEV LOSS
    # =====================================================

    if best_params is not None:

        params_enc = best_params

        print("\nMejores parámetros cargados")
        print(f"Best Dev Loss: {best_dev_loss:.6f}")




    end_time_entrenamiento = time.time()
    tiempo_entrenamiento = end_time_entrenamiento - start_time_entrenamiento

    start_time_reconstruccion = time.time()

    with torch.no_grad():

        for f in files_test:
            mse_por_bloque = [] # Lista para almacenar los MSE por bloque durante la reconstrucción

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


            if len(img_array.shape) == 3:
                img_array = img_array.squeeze(0)


            for i in range(0, resize_dim[0], block_size):
                for j in range(0, resize_dim[1], block_size):
                    block = img_array[i:i+block_size, j:j+block_size]

                    # Dividir entre 255
                    block = block / 255.0

                    state_sin_norm, block_sum, norm = funciones_estado.imagen_flatten_batch(block,device)

                    if norm > 1e-12:
                        state = state_sin_norm / norm
                    else:
                        state = state_sin_norm  # Normalizar el estado si la norma es suficientemente grande, de lo contrario usar el estado sin normalizar para evitar división por cero

                    if basis == "True" and norm > 1e-12:
                        state = (state > 0.5).int()
                       
                    
                    
                    z_vals_encoder = circuit_encoder_probs(state, params_enc, n_qubits_utiles, n_qubits_total, basis, ansatz, mejora)
                    
                    if isinstance(z_vals_encoder, list):
                        z_vals_encoder = torch.stack(z_vals_encoder)

                    row_idx = (i // block_size) * output_block_size_height
                    col_idx = (j // block_size) * output_block_size_width

                    z_vals_encoder = (1 - z_vals_encoder) / 2

                    output = funciones_estado.escalar_generar_imagen_mediciones_encoder(z_vals_encoder, output_block_size_height, output_block_size_width, norm)

                    compressed_img_small[row_idx:row_idx+output_block_size_height,
                                    col_idx:col_idx+output_block_size_width] = output

                    # Decoder 

                    z_vals_decoder = autoencoder_recon_dagger(state, params_enc, n_qubits_utiles, n_qubits_total, basis, ansatz, mejora)

                    if isinstance(z_vals_decoder, list):
                        z_vals_decoder = torch.stack(z_vals_decoder)

                    z_vals_decoder = (1 - z_vals_decoder) / 2

                    output_dec = funciones_estado.escalar_generar_imagen_mediciones_decoder(z_vals_decoder, block_size, norm)
                    reconstructed_img_small[i:(i+block_size), 
                                        j:(j+block_size)] = output_dec
                
                    if(block_sum.item() == 0.0):
                        mse = 0.0
                    else:
                        #MSE SIN INTENSDIAD PIXELES
                        z_vals_decoder = z_vals_decoder.flatten()
                        mse = torch.mean((state.flatten() - z_vals_decoder)**2).item()

                    mse_por_bloque.append(mse)

            # Guardar las imágenes resultantes
            Image.fromarray(compressed_img_small.astype(np.uint8)).save(compressed_save_path)
            Image.fromarray(img_array.detach().cpu().numpy().astype(np.uint8)).save(img_save_path)    
            Image.fromarray(reconstructed_img_small.astype(np.uint8)).save(reconstructed_save_path)


            funciones_estado.graficar(img_array, resize_dim, compressed_img_small, compressed_dim, reconstructed_img_small, combined_save_path)
            
        # ------------------ Métricas ------------------

            #Media de MSE global

            mse_nonzero = [x for x in mse_por_bloque if x != 0.0]
            media_MSE = sum(mse_nonzero) / len(mse_nonzero) if mse_nonzero else 1.0


            original = img_array.detach().cpu().numpy().astype(np.uint8)
            reconstructed = reconstructed_img_small.astype(np.uint8)
            ssim_val = ssim(original, reconstructed, data_range=255)

            mse_por_imagen.append(media_MSE) # Guardar el MSE medio de la imagen completa en la lista correspondiente para luego calcular la media global de MSE por imagen
            ssim_por_imagen.append(ssim_val) # Guardar el SSIM de la imagen completa en la lista correspondiente para luego calcular la media global de SSIM por imagen

            mse_por_clase[numero].append(media_MSE)
            mse_por_bloque = []

    end_time_reconstruccion = time.time()
    tiempo_reconstruccion = end_time_reconstruccion - start_time_reconstruccion

    start_time_analisis = time.time()




    #obtner media de MSE por todas las imagenes
    media_final_MSE = sum(mse_por_imagen) / len(mse_por_imagen) if mse_por_imagen else 0.0
    media_final_SSIM = sum(ssim_por_imagen) / len(ssim_por_imagen ) if ssim_por_imagen else 0.0



    ruta_log = os.path.join(
        base_dir,
        "Resultados",
        dataset,
        ansatz,
        mejora_str,
        carpeta_intermedia,
        f"log.txt"
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

    #GRAFICAS
    save_dir = os.path.dirname(output_dir)
    funciones_estado.get_train_dev_loss_plot(train_loss_history, dev_loss_history, save_dir)
    funciones_estado.get_batch_loss(batch_loss_history, save_dir)
    funciones_estado.get_gradient_plot(grad_history, save_dir)
    funciones_estado.get_loss_all_epoch_plot(block_loss_history_all_epochs, save_dir)

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
        num_epochs,
        tasa_de_aprendizaje, 
        n_layers,
        n_test * 10,
        ansatz,
        media_final_MSE,
        media_final_SSIM,
    )

    funciones_estado.guardar_tiempos(ruta_tiempos, 
        tiempo_total, 
        tiempo_analisis, 
        tiempo_preproceso, 
        tiempo_entrenamiento, 
        tiempo_reconstruccion
    )

