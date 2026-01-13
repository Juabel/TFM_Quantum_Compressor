from pennylane import numpy as np
from skimage.metrics import structural_similarity as ssim
import pennylane as qml
import matplotlib.pyplot as plt
import os


def imagen_flatten(img_array, i , j, block_size):
    block = img_array[i:i+block_size, j:j+block_size].astype(np.float64)

    block_norm = block / 255.0  # valores en [0,1]

    block_sum = np.sum(block_norm) / (block_size * block_size)  # Intensidad promedio del bloque original
    # Escalar promedio a [0,2]
    scaled_block_sum = block_sum * 2  # 0=negro, 2=blanco


    block_flat = block_norm.flatten(order='F')

    # Normalización L2 (para amplitude encoding)
    norm_l2 = np.linalg.norm(block_flat)

    if norm_l2 == 0:
        # Bloque negro: no hacemos nada, dejamos todos ceros
        block_flat_l2 = block_flat
    else:
        # Normalización L2
        block_flat_l2 = block_flat / norm_l2

    return block_flat_l2, block_norm, scaled_block_sum
# block_flat_l2 = El bloque normalizado en 255 y luego en L2 para usar en el encoder
# block_norm = Bloque normalizado en 255 luego usado para la comparacion (esto a revisar)
# block_sum = Intensidad bloque original que luego uso en decoder (a revisar)

def fidelidad_optima_dec(first_state, dev_dec, n_qubits, state, params_rot, tecnica_de_decoding_ansatz):
    import circuito
    final_state = circuito.create_circuit_module_dec(dev_dec, n_qubits, state, params_rot, tecnica_de_decoding_ansatz)
    overlap = qml.math.vdot(first_state, final_state) # Calcular el solapamiento entre el estado inicial y el final
    fidelidad = qml.math.abs(overlap)**2 # Calcular la fidelidad final del bloque
    return fidelidad

def medicion(dev, n_qubits, state, params_rot, tecnica_de_encoding_ansatz, block_sum):
    import circuito
    circuit = circuito.create_circuit_meas(dev, n_qubits, tecnica_de_encoding_ansatz)
    z_vals = circuit(state, params_rot)

    z_vals_scaled = np.array([((-z + 1)/2 * block_sum) for z in z_vals])
    
    # Limitar a 0-1 por seguridad (opcional, si block_sum puede ser >1)
    z_vals_scaled = np.clip(z_vals_scaled, 0, 1)

    # Convertir a rango [0,255] y limitar
    z_vals_255 = np.clip(z_vals_scaled * 255, 0, 255)
    return z_vals_255

def medicion_decoder(dev_dec, n_qubits, z_vals, params_dec, tecnica_de_decoding_ansatz, block_sum):
    import circuito
    probs = circuito.create_circuit_module_dec(dev_dec, n_qubits, z_vals, params_dec, tecnica_de_decoding_ansatz) 
    # Multiplicar por block_sum y limitar a 1
    probs = np.clip(probs * block_sum, 0, 1)
    return probs

def escalar_generar_imagen_mediciones_encoder(probs, block_sum):
    return reconstruccion_bloque_encoder(probs)  # forma original, ej: 2x2 o 4x4

def escalar_generar_imagen_mediciones_decoder(probs):
    img = reconstruccion_bloque_decoder(probs)
    # Pasar a 0-255 y convertir a uint8
    img_255 = np.clip(img * 255, 0, 255).astype(np.uint8)

    return img_255



def loss_autoencoder_block(alpha, betta, block_flat, probs):
    #MSE
   
    reconstructed = reconstruccion_bloque_decoder(probs)

    loss_val = alpha * qml.numpy.mean((block_flat - reconstructed) ** 2)
    #print("LOSS VAL :", loss_val)

    return loss_val

def mse_autoencoder_block(block_flat, probs):
    reconstructed = reconstruccion_bloque_decoder(probs)

    mse = np.mean((block_flat - reconstructed)**2)

    return mse

def reconstruccion_bloque_encoder(z_vals):
    # z_vals: (2,)
    return qml.numpy.reshape(z_vals, (2, 2), order='F')


def reconstruccion_bloque_decoder(z_vals):
    # z_vals: (4,)
    return qml.numpy.reshape(z_vals, (4, 4), order='F')

def ssim_autoencoder_block(block_flat, z_vals_decoder):

    reconstructed = reconstruccion_bloque_decoder(z_vals_decoder)

    data_range = 1.0  # ambos están en [0,1]

    return ssim(
        block_flat,
        reconstructed,
        data_range=data_range,
        win_size=3
    )


def ssim_approx(original, reconstructed):
    C1 = 1e-4
    C2 = 9e-4
    
    mu_x = np.mean(original)
    mu_y = np.mean(reconstructed)
    sigma_x = np.var(original)
    sigma_y = np.var(reconstructed)
    sigma_xy = np.mean((original - mu_x) * (reconstructed - mu_y))
    
    ssim_val = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2))
    return ssim_val

def optimizar_autoencoder_bloque(alpha, betta, dev, dev_dec, n_qubits, opt_enc, opt_dec, params_enc, params_dec, state, block_flat, tecnica_enc, tecnica_dec, block_sum):

    def loss_fn(p_enc, p_dec):
        # Encoder
        z_vals = medicion(
            dev, n_qubits, state, p_enc, tecnica_enc, block_sum
        )

        # Decoder
        probs = medicion_decoder(
            dev_dec, n_qubits, z_vals, p_dec, tecnica_dec, block_sum
        )


        #ESTOS VALORES SON COMO PROBABILIDADES, ASI QUE IGUAL HAY QUE MODIFICARLOS PARA COMPARAR CON IMAGEN ORIGINAL

        # Loss
        return loss_autoencoder_block(alpha, betta, block_flat, probs)

    # Paso de optimización

    params_enc_old = params_enc.copy()
    params_dec_old = params_dec.copy()

    params_enc = opt_enc.step(lambda p: loss_fn(p, params_dec_old), params_enc)
    params_dec = opt_dec.step(lambda p: loss_fn(params_enc_old, p), params_dec)

    return params_enc, params_dec


# def graficar(img_array, resize_dim, compressed_img_small, compressed_dim, reconstructed_img_small, show_values=False):
#     """
#     Graficar original, comprimida y reconstruida.
#     Si show_values=True, se muestran los valores de los pixeles en cada imagen.
#     """
#     fig, axs = plt.subplots(1, 3, figsize=(12, 4))

#     images = [img_array, compressed_img_small, reconstructed_img_small]
#     titles = [
#         f"Original {resize_dim[0]}x{resize_dim[1]}",
#         f"Comprimida {compressed_dim[0]}x{compressed_dim[1]}",
#         f"Reconstruida {resize_dim[0]}x{resize_dim[1]}"
#     ]

#     for ax, img, title in zip(axs, images, titles):
#         ax.imshow(img, cmap="gray", interpolation='none')
#         ax.set_title(title)
#         ax.axis('on')  # activar ejes para ver coordenadas de los pixeles
#         ax.set_xticks(np.arange(img.shape[1]))  # ticks x por pixel
#         ax.set_yticks(np.arange(img.shape[0]))  # ticks y por pixel
#         ax.set_xticklabels(np.arange(img.shape[1]))
#         ax.set_yticklabels(np.arange(img.shape[0]))
#         ax.tick_params(axis='both', which='both', length=0)  # quitar marcas largas

#         if show_values:
#             # Poner los valores de los píxeles encima
#             for i in range(img.shape[0]):
#                 for j in range(img.shape[1]):
#                     ax.text(j, i, f"{img[i, j]:.0f}",
#                             ha="center", va="center", color="red", fontsize=8)

#     plt.tight_layout()
#     plt.show()


    #------------------ Gráfica de fidelidades totales ------------------
    #plt.figure(figsize=(10, 5))
    #plt.plot(log_fidelidades_total, color='blue')
    #plt.xlabel("Iteración")
    #plt.ylabel("Fidelidad")
    #plt.title("Evolución de la fidelidad durante todo el entrenamiento")
    #plt.grid(True)
    #plt.show()

def graficar(img_array, resize_dim, compressed_img_small, compressed_dim, reconstructed_img_small):
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
    plt.show()




def guardar_log(ruta_log, resize_dim, compressed_dim, block_size, n_qubits, tecnica_de_encoding_ansatz, dataset, num_de_imagenes, tiempo_total, average_disk_ratio, average_tamaño_original, average_tamaño_comprimido):
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
        f.write(f"Ratio de compresion: {average_disk_ratio:.2f}\n")
        f.write(f"Tamano original medio: {average_tamaño_original:.6f} MB\n")
        f.write(f"Tamano comprimido medio: {average_tamaño_comprimido:.6f} MB\n")



def obtener_tamaño(path):
    return os.path.getsize(path) / (1024 ** 2)

def graficar_MSE_SSIM(train_iter_history, train_mse_history, train_ssim_history):

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(train_iter_history, train_mse_history)
    plt.xlabel("Iteración de entrenamiento")
    plt.ylabel("MSE")
    plt.title("Convergencia del MSE")
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(train_iter_history, train_ssim_history)
    plt.xlabel("Iteración de entrenamiento")
    plt.ylabel("SSIM")
    plt.title("Convergencia del SSIM")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
