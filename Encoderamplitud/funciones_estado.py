import torch.nn.functional as F
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import os
import torch
import pennylane.numpy as qnp



# Inicializamos los módulos (reutilizables)

def imagen_flatten(img_array, i, j, block_size, device):
    block = torch.tensor(img_array[i:i+block_size, j:j+block_size],
                         dtype=torch.float32, device=device)

    block_norm = block / 255.0  # [0,1]

    block_sum = block_norm.mean()

    scaled_block_sum = block_sum * 2  # ahora valores <1 → baja intensidad, >1 → alta intensidad


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

def escalar_generar_imagen_mediciones_encoder(probs, block_sum, output_block_size_height, output_block_size_width):
    return reconstruccion_bloque_encoder(probs, block_sum, output_block_size_height, output_block_size_width)  # forma original, ej: 2x2 o 4x4



def loss_autoencoder_block(block_norm, z_vals, output_block_size_height, output_block_size_width, n_bins=4, eps=1e-12):
    # Reconstrucción

    reconstructed = reconstruccion_bloque(z_vals, output_block_size_height, output_block_size_width).to(dtype=block_norm.dtype)

    p = block_norm.flatten()
    q = reconstructed.flatten()

    # Normalizar como distribución de energía
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)

    # Momentos
    mean_p = p.mean()
    mean_q = q.mean()

    var_p = p.var()
    var_q = q.var()

    # Entropía (suave)
    H_p = -(p * torch.log(p + eps)).sum()
    H_q = -(q * torch.log(q + eps)).sum()

    loss = (
        (mean_p - mean_q) ** 2 +
        (var_p - var_q) ** 2 +
        (H_p - H_q) ** 2
    )

    return loss


def reconstruccion_bloque_encoder(z_vals, block_sum, output_block_size_height, output_block_size_width):
    if not torch.is_tensor(block_sum):
        block_sum = torch.tensor(block_sum, dtype=z_vals.dtype)


    z_vals = torch.tensor(z_vals, dtype=torch.float32)  # <<< convierte la lista a tensor

    # z_vals: (2,)
    z_vals_scaled = (1 - z_vals) / 2
    z_vals255 = z_vals_scaled * 255.0

    z_vals_scaled_255 = z_vals255 * block_sum


    # return qml.numpy.reshape(z_vals_255, (output_block_size_height, output_block_size_width), order='F')
    return z_vals_scaled_255.reshape(output_block_size_height, output_block_size_width)

def reconstruccion_bloque(z_vals, output_block_size_height, output_block_size_width):
    # z_vals: (4,)
    # return z_vals.reshape(block_size, block_size)
    z_vals = (1 - z_vals) / 2  # Escalado a [0,1]

    return z_vals.reshape(output_block_size_height, output_block_size_width).T
    #return qml.numpy.reshape(z_vals, (4, 4), order='F')


def optimizar_autoencoder_bloque(opt, params, state, block_norm, circuit_enc, output_block_size_height, output_block_size_width):
    opt.zero_grad()

    params_enc = params
    # ---------- Encoder ----------
    z_vals = medicion(state, params_enc, circuit_enc)

    # ---------- Loss ----------
    loss = loss_autoencoder_block(
        block_norm,
        z_vals,
        output_block_size_height, output_block_size_width
    )

    # ---------- Backprop ---------

    loss.backward()
    opt.step()

    return (params_enc), loss.item()


def inicializar_circuitos(dev, n_qubits, tecnica_enc):
    import circuito
    circuit_enc = circuito.create_circuit_meas(dev, n_qubits, tecnica_enc)
    return circuit_enc


def graficar(img_array, resize_dim, compressed_img_small, compressed_dim):
    plt.figure(figsize=(12, 4)) # Imagen original
    plt.subplot(1, 2, 1)
    plt.imshow(img_array, cmap="gray")
    plt.title(f"Original {resize_dim[0]}x{resize_dim[1]}")
    plt.axis("off")
    # Imagen comprimida
    plt.subplot(1, 2, 2)
    plt.imshow(compressed_img_small, cmap="gray")
    plt.title(f"Comprimida {compressed_dim[0]}x{compressed_dim[1]}")
    plt.axis("off")



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


def prueba(img_array, resize_dim, compressed_img_small, compressed_dim, block_size):
    plt.figure(figsize=(12, 4))

    # Original
    plt.subplot(1, 2, 1)
    mostrar_imagen_con_bloques(
        img_array,
        f"Original {resize_dim[0]}x{resize_dim[1]}",
        block_size
    )

    # Comprimida
    plt.subplot(1, 2, 2)
    mostrar_imagen_con_bloques(
        compressed_img_small,
        f"Comprimida {compressed_dim[0]}x{compressed_dim[1]}",
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
