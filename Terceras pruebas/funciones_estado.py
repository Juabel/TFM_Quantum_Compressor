import matplotlib.pyplot as plt
import os
import torch
from sklearn.manifold import TSNE
import numpy as np
from PIL import Image, ImageFile
import time
from torch.utils.data import Dataset # What we want to learn
import random


# Permite cargar imágenes parcialmente truncadas
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Inicializamos los módulos (reutilizables)

# def imagen_flatten_batch(block, device):

#     block = block.to(dtype=torch.float32, device=device)

#     block_flat = block.reshape(-1)
#     block_sum = block_flat.sum()
#     norm = block_flat.norm()

#     return block_flat, block_sum, norm



#ANTIGUA
def imagen_flatten_batch(block, device): 
    block = block.to(dtype=torch.float32, device=device) 
    block_flat = block.reshape(-1) 
    block_sum = block_flat.sum() 
    norm = block_flat.norm() 

    # print("Bloque sin normalizar : ", block_flat)
    # print("Norma del bloque : ", norm )

    return block_flat, block_sum, norm




def medicion(state, params_encoder, circuit_enc):
    z_vals_list = circuit_enc(state, params_encoder)  # ← SIN np.array
    z_vals = torch.stack(z_vals_list)
    return z_vals

def medicion_decoder(z_vals, params_dec, circuit_dec):

    z_vals_tensor = z_vals.flatten()
    probs_list = circuit_dec(z_vals_tensor, params_dec)
    z_vals = torch.stack(probs_list)

    return z_vals

def escalar_generar_imagen_mediciones_encoder(probs, output_block_size_height, output_block_size_width, norm):
    return reconstruccion_bloque_encoder(probs, output_block_size_height, output_block_size_width, norm)  # forma original, ej: 2x2 o 4x4

def escalar_generar_imagen_mediciones_decoder(probs, block_size, norm):
    # if not torch.is_tensor(block_sum):
    #     block_sum = torch.tensor(
    #         block_sum, dtype=probs.dtype, device=probs.device
    #     )
    # Reconstrucción del bloque
    img = reconstruccion_bloque_decoder(probs, block_size)

    # Pasar a [0,255]


    img = img * norm
    img_255 = img * 255.0


    # ⚠️ SOLO para visualización (rompe gradiente)

    img_255 = torch.clamp(img_255, min=0.0, max=255.0)
    img_255_uint8 = img_255.to(torch.uint8)
    # img_255_uint8 = img_255_uint8 * block_sum


    return img_255_uint8


def reconstruccion_bloque_encoder(z_vals, output_block_size_height, output_block_size_width, norm):
    
    z_vals = z_vals.clone().detach()
    
    # print("Valores de medición (encoder) : ", z_vals)


    z_vals = z_vals * norm
        # print("Valores de medición (encoder) con factor normalizacion : ", z_vals)

    z_vals_scaled_255 = z_vals * 255.0
    # print("Valores de medición (encoder) con factor normalizacion y escalado a 255 : ", z_vals_scaled_255)

    z_vals_scaled_255 = torch.clamp(z_vals_scaled_255, min=0.0, max=255.0)
    z_vals_scaled_255 = z_vals_scaled_255.to(torch.uint8)

    return z_vals_scaled_255.reshape(output_block_size_height, output_block_size_width)

def reconstruccion_bloque_decoder(z_vals, block_size):
    # z_vals: (4,)
    # return z_vals.reshape(block_size, block_size)
    # z_vals = ((z_vals) + 1) / 2  # Escalado a [0,1]

    # return z_vals.reshape(block_size, block_size).T
    return z_vals.reshape(block_size, block_size)
    #return qml.numpy.reshape(z_vals, (4, 4), order='F')


def autoencoder_bloque(params_enc, state, autoencoder_circuit_dagger, denseAngle, ansatz, mejora):


    recon = apply_autoencoder_recon_ampl(state, params_enc, autoencoder_circuit_dagger, denseAngle, ansatz, mejora)


    # ---------- Loss ----------
    loss, recon_loss = loss_autoencoder_ampl(state, recon, denseAngle)
    # ---------- Backprop ---------

    return loss, recon_loss


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



def guardar_log(ruta_log,resize_dim,compressed_dim,block_size,n_qubits,tecnica_de_encoding_ansatz,dataset,num_de_imagenes,tiempo_total,num_epochs,tasa_aprendizaje,num_layers_decoder,num_de_imagenes_test_por_numero,ansatz,media_final_MSE,media_final_SSIM):

    carpeta = os.path.dirname(ruta_log)

    while True:
        try:
            os.makedirs(carpeta, exist_ok=True)
            print(os.listdir(carpeta))
            with open(ruta_log, "w", encoding="utf-8") as f:
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

                f.write(f"Número de epochs {num_epochs}\n")
                f.write(f"Tasa de aprendizaje: {tasa_aprendizaje}\n")
                f.write(f"Número de capas del decoder: {num_layers_decoder}\n")

                f.write(f"------------- METRICAS -------------\n")

                f.write(f"\nMSE medio global de la imagen: {media_final_MSE:.6f}\n")
                f.write(f"SSIM medio de las imagenes: {media_final_SSIM:.6f}\n")

            return

        except (FileNotFoundError, OSError) as e:
            print(f"Error guardando log: {e}")
            print("Reintentando en 30 segundos...")
            time.sleep(30)


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




def apply_autoencoder_recon_ampl(state, params_encoder, autoencoder_recon_ampl, denseAngle, ansatz, mejora):

    expvals = autoencoder_recon_ampl(state, params_encoder, denseAngle, ansatz, mejora)

    # print(qml.draw(autoencoder_recon_ampl)(state, params_encoder, params_decoder, denseAngle))


    return expvals


def loss_autoencoder_ampl(state, recon, denseAngle):
    import circuito
    if denseAngle == "True":
        recon = (1 - torch.stack(recon)) / 2
    else:
        recon = torch.sqrt(recon)
    state = state.flatten()

    loss, recon_loss = circuito.loss_autoencoder_circuito_ampl(state, recon)
    return loss, recon_loss











#################### FUNCIONES USADAS PARA ANGLE ENCODING ####################


def optimizar_autoencoder_bloque_angle(params_enc, state, autoencoder_recon_dagger, n_qubits_utiles, n_qubits_total, basis, ansatz, mejora):

    # ---------- Autoencoder ----------
    recon_expvals = apply_autoencoder_recon(state, params_enc, autoencoder_recon_dagger, n_qubits_utiles, n_qubits_total, basis, ansatz, mejora)

    # ---------- Loss ----------
    loss, recon_loss = loss_autoencoder(state, recon_expvals) 

    # ---------- Backprop ---------

    return loss, recon_loss


def apply_autoencoder_recon(state, params_encoder, autoencoder_recon, n_qubits_utiles, n_qubits_total, basis, ansatz, mejora):

    expvals = autoencoder_recon(state, params_encoder, n_qubits_utiles, n_qubits_total, basis, ansatz, mejora)  # ← SIN np.array

    return expvals



def inicializar_autoencoder_dagger(dev):
    import circuito
    autoencoder = circuito.create_autoencoder_recon_dagger(dev)
    return autoencoder

def loss_autoencoder(state, expvals):
    import circuito
    pixels_expvals = (1 - torch.stack(expvals)) / 2
    # pixels_expvals = torch.sqrt(expvals)
    block_norm = state.flatten()

    loss, recon_loss = circuito.loss_autoencoder_circuito(block_norm, pixels_expvals)
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

def optimizar_autoencoder_bloque_SQ_orig(theta, phi, params_dec, state, autoencoder_circuit, ansatz, mejora):


    output = apply_autoencoder_recon_SQ(state, theta, phi, params_dec, autoencoder_circuit, ansatz, mejora)


    # ---------- Loss ----------
    loss = loss_autoencoder_SQ(state, output)
    # ---------- Backprop ---------


    return loss, loss



def apply_autoencoder_recon_SQ(state, theta, phi, params_dec, autoencoder_recon, ansatz, mejora):

    output = autoencoder_recon(state, theta, phi, params_dec, ansatz, mejora)

    return output


def loss_autoencoder_SQ(state, recon):
    import circuito
    recon = (1 - torch.stack(recon)) / 2
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


def get_valid_blocks(img, block_size):
    threshold = 1e-6
    H, W = img.shape[-2], img.shape[-1]

    coords = []
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):

            block = img[:, i:i+block_size, j:j+block_size]

            if block.sum().item() > threshold:
                coords.append((i, j))

    return coords

def get_train_dev_loss_plot(train_loss_history,dev_loss_history,save_dir):

    plt.figure(figsize=(10, 6))

    plt.plot(
        train_loss_history,
        label="Train Loss"
    )

    plt.plot(
        dev_loss_history,
        label="Dev Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Train vs Dev Loss")

    plt.legend()

    plt.grid(True)

    save_path = os.path.join(
        save_dir,
        "train_dev_loss.png"
    )

    plt.savefig(save_path)

    plt.close()

def get_batch_loss(batch_loss_history, save_dir):
    plt.figure(figsize=(10,5))
    plt.plot(batch_loss_history)
    plt.xlabel("Batch iteration")
    plt.ylabel("Loss")
    plt.title("Loss por batch")
    plt.grid()
    plt.savefig(os.path.join(save_dir, "Loss por batch.png"))

def get_gradient_plot(grad_history, save_dir):
    steps = range(1, len(grad_history)+1)

    plt.figure(figsize=(10,5))

    plt.plot(steps, grad_history)

    plt.xlabel("Update step")
    plt.ylabel("Gradient norm")

    plt.title("Evolucion norma gradiente")

    plt.grid()

    plt.savefig(os.path.join(save_dir, "Evolucion norma gradiente.png"))


def get_loss_all_epoch_plot(block_loss_history_all_epochs, save_dir):
    plt.figure(figsize=(8,5))
    plt.hist(block_loss_history_all_epochs[-1], bins=20)
    plt.xlabel("Loss último bloque")
    plt.ylabel("Frecuencia")
    plt.title("Distribucion loss último bloques")
    plt.grid()
    plt.savefig(os.path.join(save_dir, "Distribucion loss último bloques.png"))
