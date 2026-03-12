import torch.nn.functional as F
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import os
import torch
import pennylane.numpy as qnp



# Inicializamos los módulos (reutilizables)

def imagen_flatten(img_array, i, j, block_size, device):
    block = img_array[i:i+block_size, j:j+block_size].to(dtype=torch.float32, device=device)

    block_norm = block / 255.0  # [0,1]

    block_sum = block_norm.mean()

    scaled_block_sum = block_sum * 2  # ahora valores <1 → baja intensidad, >1 → alta intensidad
    # scaled_block_sum = block_sum



    block_flat = block_norm.T.reshape(-1)

    # --- PROTECCIÓN CASO VECTOR CERO ---
    norm = block_flat.norm()

    if norm < 1e-12:
        block_flat = torch.ones_like(block_flat, device=device) / torch.sqrt(torch.tensor(block_flat.numel(), dtype=torch.float32))

    scaled_block_sum_torch = scaled_block_sum.detach().to(device)

    return block_flat, block_norm, scaled_block_sum_torch
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

def escalar_generar_imagen_mediciones_encoder(probs, block_sum, output_block_size_height, output_block_size_width):
    return reconstruccion_bloque_encoder(probs, block_sum, output_block_size_height, output_block_size_width)  # forma original, ej: 2x2 o 4x4

def escalar_generar_imagen_mediciones_decoder(probs, block_sum, block_size):
    if not torch.is_tensor(block_sum):
        block_sum = torch.tensor(
            block_sum, dtype=probs.dtype, device=probs.device
        )
    # Reconstrucción del bloque
    img = reconstruccion_bloque_decoder(probs, block_size)

    # Pasar a [0,255]
    img_255 = img * 255.0

    # ⚠️ SOLO para visualización (rompe gradiente)



    img_255_uint8 = img_255.to(torch.uint8)
    img_255_uint8 = img_255_uint8 * block_sum
    img_255_uint8 = torch.clamp(img_255_uint8, min=0.0, max=255.0)

    return img_255_uint8


def loss_autoencoder_block(alpha, betta, block_norm, probs, block_size):
    import circuito
    
    # Reconstrucción


    #CORREGIR Y CAMBIAR ESTO AHORA CON STRONGLYENTANGLED Y EL LOSS HACERLO EN EL CIRCUITO.PY PARA NO TENER QUE HACER UN LOSS DIFERENCIABLE




    reconstructed = reconstruccion_bloque_decoder(probs, block_size).to(dtype=block_norm.dtype)
    # Añadimos batch y canal
    # reconstructed_ = reconstructed.unsqueeze(0).unsqueeze(0)
    # block_norm_ = block_norm.unsqueeze(0).unsqueeze(0)

    # --- MSE diferenciable ---
    mse_val = circuito.loss_autoencoder_amplitude(reconstructed, block_norm)


    # --- SSIM diferenciable ---
    # ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(reconstructed_.device)
    # ssim_val = ssim_fn(reconstructed_, block_norm_)

    # --- Loss conjunta ---
    loss_val = alpha * mse_val
    # loss_val = alpha * mse_val + betta * (1 - ssim_val)

    return loss_val

def reconstruccion_bloque_encoder(z_vals, block_sum, output_block_size_height, output_block_size_width):


    if not torch.is_tensor(block_sum):
        block_sum = torch.tensor(block_sum, dtype=z_vals.dtype)


    z_vals = z_vals.clone().detach()


    # z_vals: (2,)
    z_vals_scaled = (1 - z_vals) / 2
    z_vals255 = z_vals_scaled * 255.0
    z_vals_scaled_255 = z_vals255 * block_sum
    z_vals_scaled_255 = torch.clamp(z_vals_scaled_255, min=0.0, max=255.0)



    # return qml.numpy.reshape(z_vals_255, (output_block_size_height, output_block_size_width), order='F')
    return z_vals_scaled_255.reshape(output_block_size_height, output_block_size_width)

def reconstruccion_bloque_decoder(z_vals, block_size):
    # z_vals: (4,)
    # return z_vals.reshape(block_size, block_size)




    #PROBAR

    z_vals = (1 - (z_vals)) / 2  # Escalado a [0,1]
    # z_vals = ((z_vals) + 1) / 2  # Escalado a [0,1]




    return z_vals.reshape(block_size, block_size).T
    #return qml.numpy.reshape(z_vals, (4, 4), order='F')


def optimizar_autoencoder_bloque(alpha, betta, opt, params, state, block_norm, circuit_enc, circuit_dec, block_size):

    opt.zero_grad()

    params_enc, params_dec = params
    # ---------- Encoder ----------
    z_vals = medicion(state, params_enc, circuit_enc)

    # ---------- Decoder ----------
    probs = medicion_decoder(z_vals, params_dec, circuit_dec)

    # ---------- Loss ----------
    loss = loss_autoencoder_block(
        alpha,
        betta,
        block_norm,
        probs,
        block_size
    )

    # ---------- Backprop ---------

    loss.backward()
    opt.step()

    return (params_enc, params_dec), loss.item()


def inicializar_circuitos(dev, dev_dec, n_qubits):
    import circuito
    circuit_enc = circuito.create_circuit_meas(dev, n_qubits)
    circuit_dec = circuito.create_circuit_module_dec(dev_dec, n_qubits)
    return circuit_enc, circuit_dec


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


def optimizar_autoencoder_bloque_angle(opt, params_enc, params_dec, state, autoencoder_recon, autoencoder_recon_dagger, dagger):

    opt.zero_grad()

    # ---------- Autoencoder ----------
    if dagger == "True":
        recon_expvals, trash_expvals, latent_x0, latent_y0, latent_x1, latent_y1 = apply_autoencoder_recon(state, params_enc, params_dec, autoencoder_recon_dagger)
    else:
        recon_expvals, trash_expvals, latent_x0, latent_y0, latent_x1, latent_y1 = apply_autoencoder_recon(state, params_enc, params_dec, autoencoder_recon)

    # ---------- Loss ----------
    loss, recon_loss = loss_autoencoder(state, recon_expvals, trash_expvals, lambda_trash=0.9, latent_x0=latent_x0, latent_y0=latent_y0, latent_x1=latent_x1, latent_y1=latent_y1, lambda_bloch=0.5) 

    # ---------- Backprop ---------

    loss.backward()
    opt.step()

    return params_enc, params_dec, loss.item(), recon_loss.item()






def apply_autoencoder_recon(state, params_encoder, params_decoder, autoencoder_recon):

    expvals, trash_expvals, latent_x0, latent_y0, latent_x1, latent_y1 = autoencoder_recon(state, params_encoder, params_decoder)  # ← SIN np.array

    return expvals, trash_expvals, latent_x0, latent_y0, latent_x1, latent_y1



def inicializar_autoencoder(dev):
    import circuito
    autoencoder = circuito.create_autoencoder_recon(dev)
    return autoencoder

def inicializar_autoencoder_dagger(dev):
    import circuito
    autoencoder = circuito.create_autoencoder_recon_dagger(dev)
    return autoencoder

def loss_autoencoder(block_norm, expvals, trash_expvals, lambda_trash, latent_x0, latent_y0, latent_x1, latent_y1, lambda_bloch):
    import circuito
    pixels_expvals = (1 - torch.stack(expvals)) / 2
    block_norm = block_norm.flatten()

    bloch_penalty = latent_x0**2 + latent_y0**2 + latent_x1**2 + latent_y1**2
    loss, recon_loss = circuito.loss_autoencoder_circuito(block_norm, pixels_expvals, trash_expvals, lambda_trash, bloch_penalty, lambda_bloch)
    return loss, recon_loss

def inicializa_encoder_probs(dev):
    import circuito
    circuit_enc = circuito.create_encoder_probs(dev)
    return circuit_enc

def coarse_grain_probs(vals, n_pixels_out):
    vals = vals.reshape(n_pixels_out, -1)
    return vals.sum(dim=1)