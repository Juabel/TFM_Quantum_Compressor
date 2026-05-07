import matplotlib.pyplot as plt
import os
import torch
from sklearn.manifold import TSNE
import numpy as np
from PIL import Image, ImageFile
import time

# Permite cargar imágenes parcialmente truncadas
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Inicializamos los módulos (reutilizables)

def imagen_flatten(img_array, i, j, block_size, device):
    block = img_array[i:i+block_size, j:j+block_size].to(dtype=torch.float32, device=device)

    block_norm = block / 255.0  # [0,1]

    block_sum = block_norm.mean()

    scaled_block_sum = block_sum * 2  # ahora valores <1 → baja intensidad, >1 → alta intensidad

    # block_flat = block_norm.T.reshape(-1)
    block_flat = block_norm.reshape(-1)


    

    # --- PROTECCIÓN CASO VECTOR CERO ---
    norm = block_flat.norm()

    if norm < 1e-12:
        block_flat = torch.ones_like(block_flat, device=device) / torch.sqrt(torch.tensor(block_flat.numel(), dtype=torch.float32))

    scaled_block_sum_torch = scaled_block_sum.detach().to(device)

    return block_flat, block_norm, scaled_block_sum_torch, norm
    # return block_flat, block_norm, scaled_block_sum

# block_flat = El bloque normalizado en 255 para usar en el encoder
# block_norm = Bloque normalizado en 255 luego usado para la comparacion (esto a revisar)
# block_sum = Intensidad bloque original que luego uso en decoder (a revisar)

def medicion(state, params_encoder, circuit_enc):
    z_vals_list = circuit_enc(state, params_encoder)  # ← SIN np.array
    z_vals = torch.stack(z_vals_list)
    return z_vals

def medicion_decoder(z_vals, params_dec, circuit_dec):

    z_vals_tensor = z_vals.flatten()
    probs_list = circuit_dec(z_vals_tensor, params_dec)
    z_vals = torch.stack(probs_list)

    return z_vals

def escalar_generar_imagen_mediciones_encoder(probs, output_block_size_height, output_block_size_width, norm, basis):
    return reconstruccion_bloque_encoder(probs, output_block_size_height, output_block_size_width, norm, basis)  # forma original, ej: 2x2 o 4x4

def escalar_generar_imagen_mediciones_decoder(probs, block_size, norm, basis):
    # if not torch.is_tensor(block_sum):
    #     block_sum = torch.tensor(
    #         block_sum, dtype=probs.dtype, device=probs.device
    #     )
    # Reconstrucción del bloque
    img = reconstruccion_bloque_decoder(probs, block_size)

    # Pasar a [0,255]

    if basis == "False":
        img = img * norm
    img_255 = img * 255.0


    # ⚠️ SOLO para visualización (rompe gradiente)

    img_255 = torch.clamp(img_255, min=0.0, max=255.0)
    img_255_uint8 = img_255.to(torch.uint8)
    # img_255_uint8 = img_255_uint8 * block_sum


    return img_255_uint8


def reconstruccion_bloque_encoder(z_vals, output_block_size_height, output_block_size_width, norm, basis):
    # if not torch.is_tensor(block_sum):
    #     block_sum = torch.tensor(block_sum, dtype=z_vals.dtype)


    z_vals = z_vals.clone().detach()
    # z_vals: (2,)

    if basis == "False":
        z_vals = z_vals * norm
    z_vals_scaled_255 = z_vals * 255.0

    # z_vals_scaled_255 = z_vals_scaled_255 * block_sum

    z_vals_scaled_255 = torch.clamp(z_vals_scaled_255, min=0.0, max=255.0)
    z_vals_scaled_255 = z_vals_scaled_255.to(torch.uint8)



    # return qml.numpy.reshape(z_vals_255, (output_block_size_height, output_block_size_width), order='F')
    return z_vals_scaled_255.reshape(output_block_size_height, output_block_size_width)

def reconstruccion_bloque_decoder(z_vals, block_size):
    # z_vals: (4,)
    # return z_vals.reshape(block_size, block_size)
    # z_vals = ((z_vals) + 1) / 2  # Escalado a [0,1]

    # return z_vals.reshape(block_size, block_size).T
    return z_vals.reshape(block_size, block_size)
    #return qml.numpy.reshape(z_vals, (4, 4), order='F')


def optimizar_autoencoder_bloque(opt, params, state, autoencoder_circuit, autoencoder_circuit_dagger, dagger, denseAngle, ansatz, mejora):

    opt.zero_grad()

    params_enc, params_dec = params

    if dagger == "True":
        recon, trash = apply_autoencoder_recon_ampl(state, params_enc, params_dec, autoencoder_circuit_dagger, denseAngle, ansatz, mejora)
    else:
        recon, trash, = apply_autoencoder_recon_ampl(state, params_enc, params_dec, autoencoder_circuit, denseAngle, ansatz, mejora)


    # ---------- Loss ----------
    loss, recon_loss = loss_autoencoder_ampl(state, recon, trash, denseAngle, lambda_trash=0.9)
    # ---------- Backprop ---------

    loss.backward()
    opt.step()

    return (params_enc, params_dec), loss.item(), recon_loss.item()


def graficar(img_array, resize_dim, compressed_img_small, compressed_dim, reconstructed_img_small, output_path):
    plt.figure(figsize=(12, 4)) # Imagen original
    plt.subplot(1, 3, 1)
    plt.imshow(img_array, cmap="gray")
    plt.title(f"Original {resize_dim[0]}x{resize_dim[1]}")
    plt.axis("off")
    # Imagen comprimida
    plt.subplot(1, 3, 2)
    plt.imshow(compressed_img_small, cmap="gray")
    plt.title(f"Comprimida {compressed_dim[0]}x{compressed_dim[1]}")
    plt.axis("off")
    # Imagen reconstruida
    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed_img_small, cmap="gray")
    plt.title(f"Reconstruida {resize_dim[0]}x{resize_dim[1]}")
    plt.axis("off")
    plt.tight_layout()

        # 🔹 Guardar la figura completa
    plt.savefig(output_path, bbox_inches='tight')
    
    # plt.show()



def mostrar_imagen_con_bloques(img, titulo, block_size):
    h, w = img.shape

    plt.imshow(img, cmap="gray")
    plt.title(titulo)
    plt.axis("off")

    # Líneas de bloques
    for x in range(0, w, block_size):
        plt.axvline(x - 0.5, color="cyan", linewidth=0.8)

    for y in range(0, h, block_size):
        plt.axhline(y - 0.5, color="cyan", linewidth=0.8)


def prueba(img_array, resize_dim, compressed_img_small, compressed_dim, reconstructed_img_small, block_size):
    plt.figure(figsize=(12, 4))

    # Original
    plt.subplot(1, 3, 1)
    mostrar_imagen_con_bloques(
        img_array,
        f"Original {resize_dim[0]}x{resize_dim[1]}",
        block_size
    )

    # Comprimida
    plt.subplot(1, 3, 2)
    mostrar_imagen_con_bloques(
        compressed_img_small,
        f"Comprimida {compressed_dim[0]}x{compressed_dim[1]}",
        block_size
    )

    # Reconstruida
    plt.subplot(1, 3, 3)
    mostrar_imagen_con_bloques(
        reconstructed_img_small,
        f"Reconstruida {resize_dim[0]}x{resize_dim[1]}",
        block_size
    )

    plt.tight_layout()
    plt.show()



def guardar_log(ruta_log, resize_dim, compressed_dim, block_size, n_qubits, tecnica_de_encoding_ansatz, dataset, num_de_imagenes, tiempo_total, average_disk_ratio, average_tamaño_original, average_tamaño_comprimido
                ,num_iteraciones_globales, num_iteraciones_bloque, tasa_aprendizaje, num_layers_decoder, num_de_imagenes_test_por_numero, ansatz, media_final_MSE, media_final_SSIM, media_final_MSE_intensity):
    with open(ruta_log, "w") as f:
        f.write("\n\n\n")
        f.write(f"------------- HIPERPARAMETROS -------------\n")
        f.write(f"Resize dimension: {resize_dim}\n")
        f.write(f"Compressed dimension: {compressed_dim}\n")
        f.write(f"Tamano de bloques: {block_size}x{block_size}\n")
        f.write(f"Numero de qubits total: {n_qubits}\n")
        f.write(f"Tecnica de encoding: {tecnica_de_encoding_ansatz}\n")
        f.write(f"Tipo de imagenes: {dataset}\n")
        f.write(f"Numero de imagenes total Train: {num_de_imagenes}\n")
        f.write(f"Numero de imagenes total Test: {num_de_imagenes_test_por_numero}\n")
        f.write(f"Ansatz usado: {ansatz}\n")
        f.write(f"-------------  -------------\n")

        #numero iteraciones globales, locales, tasa de aprendizaje, numero de layers de deocoder, 
        f.write(f"Número de iteraciones globales: {num_iteraciones_globales}\n")
        f.write(f"Número de iteraciones locales: {num_iteraciones_bloque}\n")
        f.write(f"Tasa de aprendizaje: {tasa_aprendizaje}\n")
        f.write(f"Número de capas del decoder: {num_layers_decoder}\n")

        f.write(f"-------------  -------------\n")
        f.write(f"Tiempo de ejecucion: {tiempo_total:.2f} segundos\n")
        f.write(f"Ratio de compresion: {average_disk_ratio:.2f}\n")
        f.write(f"Tamano original medio: {average_tamaño_original:.6f} MB\n")
        f.write(f"Tamano comprimido medio: {average_tamaño_comprimido:.6f} MB\n")
        
        f.write(f"------------- METRICAS -------------\n")

        f.write(f"\nMSE medio global de la imagen: {media_final_MSE:.6f}\n")
        f.write(f"SSIM medio de las imagenes: {media_final_SSIM:.6f}\n")
        f.write(f"MSE medio con intensidad de pixeles: {media_final_MSE_intensity:.6f}\n")


def guardar_tiempos(ruta_tiempos, tiempo_total, tiempo_analisis, tiempo_preproceso, tiempo_entrenamiento, tiempo_reconstruccion):
    with open(ruta_tiempos, "w") as f:
        f.write("\n\n\n")
        f.write(f"------------- TIEMPOS DE EJECUCION POR APARTADO -------------\n")
        f.write(f"Tiempo total de la ejecucion: {tiempo_total}\n")
        f.write(f"Tiempo preprocesado de información antes de entrenamiento: {tiempo_preproceso}\n")
        f.write(f"Tiempo entrenamiento: {tiempo_entrenamiento}\n")
        f.write(f"Tiempo reconstruccion: {tiempo_reconstruccion}\n")
        f.write(f"Tiempo analisis con graficos mas guardado de logs: {tiempo_analisis}\n")
        




def obtener_tamaño(path):
    return os.path.getsize(path) / (1024 ** 2)

def graficar_MSE(train_iter_history, train_mse_history):
    plt.figure()

    for epoch in range(len(train_mse_history)):
        plt.plot(
            train_iter_history[epoch],
            train_mse_history[epoch],
            label=f"Epoch {epoch+1}"
        )

    plt.xlabel("Iteración global")
    plt.ylabel("MSE")
    plt.title("Evolución del MSE por iteraciones globales")
    plt.legend()
    plt.grid()
    plt.show()


def histograma_MSE_por_clase(mse_por_clase, saver_dir):

    clases = []
    medias = []

    for clase in sorted(mse_por_clase.keys(), key=int):
        valores = mse_por_clase[clase]
        media = np.mean(valores) if valores else 0
        clases.append(clase)
        medias.append(media)

    plt.figure()
    plt.bar(clases, medias)
    plt.xlabel("Dígito")
    plt.ylabel("MSE medio")
    plt.title("MSE medio por clase (MNIST)")
    plt.savefig(os.path.join(saver_dir, "histograma_MSE_por_clase.png"))
    # plt.show()

def bloques_procesados(x, mse_epoch):
    plt.plot(x, mse_epoch)
    plt.xlabel("Bloques procesados")
    plt.ylabel("MSE")
    plt.title("Evolución del error durante entrenamiento")
    plt.show()


def bloques_finales(x2, train_mse_history):
    # Convertir a array (cuidado: deben tener misma longitud)
    min_len = min(len(lst) for lst in train_mse_history)
    
    recortadas = [lst[:min_len] for lst in train_mse_history]
    data = np.array(recortadas)

    media = np.mean(data, axis=0)

    std = np.std(data, axis=0)

    x = range(len(media))

    plt.plot(x, media, label="Media")
    plt.fill_between(x, media-std, media+std, alpha=0.3)

    plt.xlabel("Bloques / iteraciones")
    plt.ylabel("MSE")
    plt.title("Evolución del error (media ± std)")
    plt.legend()
    plt.show()

def comportamiento_entrenamiento_global(train_mse_imagenes, saver_dir):

    import numpy as np
    import matplotlib.pyplot as plt

    # filtrar vacíos
    clean = [np.array(img) for img in train_mse_imagenes if len(img) > 0]

    min_len = min(len(img) for img in clean)
    data = np.array([img[:min_len] for img in clean])

    mean_mse = np.mean(data, axis=0)

    # asegurar 1D real
    mean_mse = np.asarray(mean_mse).reshape(-1)

    x = np.arange(len(mean_mse))

    plt.figure()
    plt.plot(x, mean_mse, color="black", linewidth=2)

    plt.xlabel("Iteración global")
    plt.ylabel("MSE")
    plt.title("Evolución global del entrenamiento")
    plt.savefig(os.path.join(saver_dir, "evolucion_global_entrenamiento.png"))
    # plt.show()

def mapa_de_calor(error_medio, save_dir):
    plt.imshow(error_medio, cmap='gray_r', vmin=np.min(error_medio), vmax=np.max(error_medio))
    plt.colorbar()
    plt.title("Error medio (ignorando fondo)")
    plt.savefig(os.path.join(save_dir, "mapa_de_calor.png"))
    # plt.show()


def graficar_TSNE(latentes, labels, save_dir):
    perplexity = min(30, len(latentes) - 1)
    X = np.array(latentes)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(8,6))

    for i in range(10):
        idx = [j for j, l in enumerate(labels) if l == i]
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=str(i), alpha=0.6)

    plt.legend()
    plt.title("t-SNE por clases (MNIST)")
    plt.savefig(os.path.join(save_dir, "t-SNE_por_clases.png"))
    # plt.show()

def inicializar_autoencoder_amplitude(dev):
    import circuito
    autoencoder = circuito.create_autoencoder_recon_ampl(dev)
    return autoencoder
def inicializar_autoencoder_amplitude_dagger(dev):
    import circuito
    autoencoder = circuito.create_autoencoder_recon_ampl_dagger(dev)
    return autoencoder
def inicializa_encoder_probs_ampl(dev):
    import circuito
    circuit_enc = circuito.create_encoder_probs_ampl(dev)
    return circuit_enc

def inicializar_autoencoder_SQ(dev):
    import circuito
    autoencoder = circuito.create_autoencoder_recon_single(dev)
    return autoencoder

def inicializa_encoder_probs_SQ(dev):
    import circuito
    circuit_enc = circuito.create_encoder_probs_SQ(dev)
    return circuit_enc




def apply_autoencoder_recon_ampl(state, params_encoder, params_decoder, autoencoder_recon_ampl, denseAngle, ansatz, mejora):

    expvals, trash_expvals = autoencoder_recon_ampl(state, params_encoder, params_decoder, denseAngle, ansatz, mejora)

    # print(qml.draw(autoencoder_recon_ampl)(state, params_encoder, params_decoder, denseAngle))


    return expvals, trash_expvals


def loss_autoencoder_ampl(state, recon, trash, denseAngle, lambda_trash):
    import circuito
    if denseAngle == "True":
        recon = (1 - torch.stack(recon)) / 2
    else:
        recon = torch.sqrt(recon)
    state = state.flatten()

    loss, recon_loss = circuito.loss_autoencoder_circuito_ampl(state, recon, trash, lambda_trash)
    return loss, recon_loss











#################### FUNCIONES USADAS PARA ANGLE ENCODING ####################


def optimizar_autoencoder_bloque_angle(opt, params_enc, params_dec, state, autoencoder_recon, autoencoder_recon_dagger, dagger, n_qubits_utiles, n_qubits_total, basis, ansatz, mejora):

    opt.zero_grad()

    # ---------- Autoencoder ----------
    if dagger == "True":
        recon_expvals, trash_expvals = apply_autoencoder_recon(state, params_enc, params_dec, autoencoder_recon_dagger, n_qubits_utiles, n_qubits_total, basis, ansatz, mejora)
    else:
        recon_expvals, trash_expvals = apply_autoencoder_recon(state, params_enc, params_dec, autoencoder_recon, n_qubits_utiles, n_qubits_total, basis, ansatz, mejora)

    # ---------- Loss ----------
    loss, recon_loss = loss_autoencoder(state, recon_expvals, trash_expvals, lambda_trash=0.9) 

    # ---------- Backprop ---------

    loss.backward()
    opt.step()

    return params_enc, params_dec, loss.item(), recon_loss.item()


def apply_autoencoder_recon(state, params_encoder, params_decoder, autoencoder_recon, n_qubits_utiles, n_qubits_total, basis, ansatz, mejora):

    expvals, trash_expvals = autoencoder_recon(state, params_encoder, params_decoder, n_qubits_utiles, n_qubits_total, basis, ansatz, mejora)  # ← SIN np.array

    return expvals, trash_expvals


def inicializar_autoencoder(dev):
    import circuito
    autoencoder = circuito.create_autoencoder_recon(dev)
    return autoencoder

def inicializar_autoencoder_dagger(dev):
    import circuito
    autoencoder = circuito.create_autoencoder_recon_dagger(dev)
    return autoencoder

def loss_autoencoder(block_norm, expvals, trash_expvals, lambda_trash):
    import circuito
    pixels_expvals = (1 - torch.stack(expvals)) / 2
    # pixels_expvals = torch.sqrt(expvals)
    block_norm = block_norm.flatten()

    loss, recon_loss = circuito.loss_autoencoder_circuito(block_norm, pixels_expvals, trash_expvals, lambda_trash)
    return loss, recon_loss

def inicializa_encoder_probs(dev):
    import circuito
    circuit_enc = circuito.create_encoder_probs(dev)
    return circuit_enc




#################### FUNCIONES USADAS PARA SQ ####################


def optimizar_autoencoder_bloque_SQ(opt, theta, phi, params_dec, state, autoencoder_circuit):

    opt.zero_grad()

    output = apply_autoencoder_recon_SQ(state, theta, phi, params_dec, autoencoder_circuit)


    # ---------- Loss ----------
    loss = loss_autoencoder_SQ(state, output)
    # ---------- Backprop ---------

    loss.backward()
    opt.step()

    return theta, phi, params_dec, loss.item(), loss.item()

def optimizar_autoencoder_bloque_SQ_orig(opt, params, state, autoencoder_circuit, opt_params, ansatz, mejora):

    opt.zero_grad()

    theta, phi, params_dec = params

    output = apply_autoencoder_recon_SQ(state, theta, phi, params_dec, autoencoder_circuit, opt_params, ansatz, mejora)


    # ---------- Loss ----------
    loss = loss_autoencoder_SQ(state, output)
    # ---------- Backprop ---------

    loss.backward()
    opt.step()

    return (theta, phi, params_dec), loss.item(), loss.item()



def apply_autoencoder_recon_SQ(state, theta, phi, params_dec, autoencoder_recon, opt_params, ansatz, mejora):

    output = autoencoder_recon(state, theta, phi, params_dec, opt_params, ansatz, mejora)

    return output


def loss_autoencoder_SQ(state, recon):
    import circuito
    recon = torch.sqrt(recon)
    state = state.flatten()

    loss = circuito.loss_autoencoder_circuito_SQ(state, recon)
    return loss


def open_image_safe(path):
    while True:
        try:
            with Image.open(path) as img:

                img.load()
                return img.copy()
        except Exception:
            time.sleep(0.2)
    return None