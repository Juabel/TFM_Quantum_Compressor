# quantum_autoencoder_full.py
"""
Autoencoder cuántico por bloques 4x4 (16 amplitudes) con amplitude encoding.
Entrena encoder y decoder mediante alternancia:
 - decoder: optimizador con gradiente (qml.AdamOptimizer)
 - encoder: optimizador sin gradiente (scipy.minimize - Powell)
Métrica de entrenamiento: MSE sobre probabilidades (probs del decoder).
También calcula SSIM como métrica informativa.
"""

import os
import time
from PIL import Image
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ---------------------------
#  Configuración (ajustable)
# ---------------------------
IMAGE_PATH = "SAR_Dataset/0/3.png"   # pon aquí tu imagen (grayscale preferiblemente)
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

BLOCK_SIZE = 4   # 4x4 blocks -> 16 amplitudes -> 4 qubits
N_QUBITS = 4

# Optimización
DECODER_STEPS = 40         # pasos de Adam para el decoder por iteración
ENCODER_MAXITER = 120      # iteraciones de Powell para el encoder (por bloque optimizado)
ALTERNATE_STEPS = 8        # cuántas alternancias encoder<->decoder por bloque
LEARNING_RATE_DEC = 0.05

# Ansatze profundidad
ENC_LAYERS = 2
DEC_LAYERS = 2

# ---------------------------
#  Dispositivos y QNodes
# ---------------------------
dev_enc = qml.device("default.qubit", wires=N_QUBITS)
dev_dec = qml.device("default.qubit", wires=N_QUBITS)

def encoder_ansatz(params):
    """Simple ansatz: capas RY por qubit + CNOT cadena"""
    params = pnp.reshape(params, (ENC_LAYERS, N_QUBITS))
    for l in range(ENC_LAYERS):
        for q in range(N_QUBITS):
            qml.RY(params[l, q], wires=q)
        for q in range(N_QUBITS - 1):
            qml.CNOT(wires=[q, q+1])

def decoder_ansatz(params):
    """Ansatz simétrico al encoder (pero con parámetros propios)"""
    params = pnp.reshape(params, (DEC_LAYERS, N_QUBITS))
    for l in range(DEC_LAYERS):
        for q in range(N_QUBITS):
            qml.RY(params[l, q], wires=q)
        for q in range(N_QUBITS - 1):
            qml.CNOT(wires=[q, q+1])

@qml.qnode(dev_enc, interface="autograd", diff_method="parameter-shift")
def qnode_encoder(amplitudes, params_enc):
    """
    Entrada: amplitudes (16 longs) normalizadas (||amplitudes||2 == 1)
    Salida: 4 expectativas PauliZ -> latente clásico
    """
    qml.AmplitudeEmbedding(amplitudes, wires=range(N_QUBITS), normalize=False)
    encoder_ansatz(params_enc)
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


@qml.qnode(dev_dec, interface="autograd", diff_method="parameter-shift")
def qnode_decoder(latent, params_dec):
    """
    Entrada: latent (4 valores en [-1,1]) -> los reinyecta con AngleEmbedding
    Salida: probabilidades (16 valores)
    """
    # AngleEmbedding espera vectores; mapeamos latent a [0, 2pi] para rotaciones
    angles = (latent + 1.0) * np.pi  # [-1,1] -> [0,2pi]
    qml.templates.AngleEmbedding(angles, wires=range(N_QUBITS), rotation="Y")
    decoder_ansatz(params_dec)
    return qml.probs(wires=range(N_QUBITS))   # vector length 16

# ---------------------------
#  Utilidades: manipular bloques
# ---------------------------
def image_to_blocks(img_array, block_size):
    """Convierte imagen 2D en lista de bloques (flattened Fortran order)"""
    h, w = img_array.shape
    blocks = []
    coords = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img_array[i:i+block_size, j:j+block_size]
            if block.shape != (block_size, block_size):
                # ignorar bordes si no encajan exactamente
                continue
            block_flat = block.flatten(order='F').astype(float)  # Fortran ordering
            blocks.append(block_flat)
            coords.append((i, j))
    return blocks, coords

def normalize_amplitudes(block_flat):
    """Normaliza block_flat a amplitudes (||a||2==1). Si todo cero, devuelve estado |0...>"""
    norm = np.linalg.norm(block_flat)
    if norm == 0:
        amps = np.zeros_like(block_flat)
        amps[0] = 1.0
    else:
        amps = block_flat / norm
    return amps, norm

def block_target_probs(block_flat):
    """Target probabilities para comparar con qnode_decoder: |amps|^2"""
    amps, norm = normalize_amplitudes(block_flat)
    probs = np.abs(amps)**2
    return probs, norm

# ---------------------------
#  Funciones de pérdida / entrenamiento
# ---------------------------
def mse_loss(probs, target_probs):
    return np.mean((probs - target_probs)**2)

def train_block(block_flat, init_params_enc=None, init_params_dec=None, verbose=False):
    """
    Entrena encoder y decoder para un bloque (bloque = vector 16).
    Devuelve params_enc, params_dec, history (mse, ssim)
    """
    # targets
    target_probs, norm = block_target_probs(block_flat)

    # Inicializar parámetros si necesario
    enc_param_count = ENC_LAYERS * N_QUBITS
    dec_param_count = DEC_LAYERS * N_QUBITS
    if init_params_enc is None:
        params_enc = 0.1 * np.random.randn(enc_param_count)
    else:
        params_enc = init_params_enc.copy()
    if init_params_dec is None:
        params_dec = 0.1 * np.random.randn(dec_param_count)
    else:
        params_dec = init_params_dec.copy()

    # Optimizador para decoder (tiene gradiente directo)
    dec_opt = qml.AdamOptimizer(stepsize=LEARNING_RATE_DEC)

    history = {"mse": [], "ssim": []}

    # Alternancia encoder <-> decoder
    for alt in range(ALTERNATE_STEPS):
        if verbose:
            print(f" Block alt {alt+1}/{ALTERNATE_STEPS}")

        # --- 1) Optimizar decoder manteniendo encoder fijo ---
        for step in range(DECODER_STEPS):
            # obtener latente actual (clásico)
            amps, _ = normalize_amplitudes(block_flat)
            latent = np.array(qnode_encoder(amps, params_enc), requires_grad=False)  # 4 valores
            # step of Adam: dec_opt.step expects a function returning loss
            def dec_loss_wrap(p_dec):
                probs = qnode_decoder(latent, p_dec)
                return mse_loss(probs, target_probs)
            params_dec = dec_opt.step(dec_loss_wrap, params_dec)
            if (step + 1) % 20 == 0 and verbose:
                cur_loss = dec_loss_wrap(params_dec)
                print(f"   Decoder step {step+1}, loss={cur_loss:.6f}")

        # --- 2) Optimizar encoder usando un método sin gradiente (Powell) ---
        # Defino función objetivo que, dado params_enc (vector), calcula la loss
        def encoder_obj_flat(x_enc):
            # x_enc es numpy 1D
            amps, _ = normalize_amplitudes(block_flat)
            latent = np.array(qnode_encoder(amps, x_enc), requires_grad=False)
            probs = qnode_decoder(latent, params_dec)
            return float(mse_loss(probs, target_probs))

        # Llamada a optimize.minimize (Powell)
        res = minimize(encoder_obj_flat, params_enc, method="Powell",
                       options={"maxiter": ENCODER_MAXITER, "disp": False})
        params_enc = res.x

        # --- Registrar métricas actuales ---
        amps, _ = normalize_amplitudes(block_flat)
        latent = np.array(qnode_encoder(amps, params_enc), requires_grad=False)
        probs = qnode_decoder(latent, params_dec)
        loss_now = mse_loss(probs, target_probs)

        # Para SSIM necesitamos una imagen 2D; usamos reconstrucción transformada:
        # reconstrucción en intensidades aproximada: amps_recon = sqrt(probs) * norm
        amps_recon = np.sqrt(probs) * (np.linalg.norm(block_flat) if np.linalg.norm(block_flat) != 0 else 1.0)
        recon_block_2d = amps_recon.reshape((BLOCK_SIZE, BLOCK_SIZE), order='F')
        orig_block_2d = block_flat.reshape((BLOCK_SIZE, BLOCK_SIZE), order='F')
        # Normalizar a rango 0-255 para ssim:
        # si los valores están ya en [0,255], ok; si no, los escalamos a 0-255 por su max
        # Evitamos división por 0:
        if orig_block_2d.max() == orig_block_2d.min():
            ssim_val = 1.0 if np.allclose(orig_block_2d, recon_block_2d) else 0.0
        else:
            # reescalamos ambos a 0-255
            def scale_0_255(x):
                x = x.astype(np.float64)
                x = (x - x.min()) / (x.max() - x.min())
                return (x * 255).astype(np.uint8)
            ssim_val = ssim(scale_0_255(orig_block_2d), scale_0_255(np.clip(recon_block_2d, 0, None)), data_range=255, win_size=3)

        history["mse"].append(loss_now)
        history["ssim"].append(ssim_val)
        if verbose:
            print(f" Alt {alt+1} done: MSE={loss_now:.6f}, SSIM≈{ssim_val:.4f}")

    return params_enc, params_dec, history

# ---------------------------
#  Pipeline completo: imagen -> bloques -> entrenamiento por bloque -> reconstrucción
# ---------------------------
def process_image_train(image_path, visualize=False):
    img = Image.open(image_path).convert("L")  # grayscale
    # Ajustar tamaño para que sea múltiplo de BLOCK_SIZE
    w, h = img.size
    w2 = (w // BLOCK_SIZE) * BLOCK_SIZE
    h2 = (h // BLOCK_SIZE) * BLOCK_SIZE
    if w2 != w or h2 != h:
        img = img.crop((0, 0, w2, h2))
    img_array = np.array(img).astype(float)

    blocks, coords = image_to_blocks(img_array, BLOCK_SIZE)

    # Preparar arrays de resultados
    reconstructed = np.zeros_like(img_array)
    compressed_small = np.zeros((h2 // 2, w2 // 2))  # porque cada bloque 4x4 -> 2x2 compressed (4 vals)
    all_histories = []

    t0 = time.time()
    for idx, (block_flat, (i, j)) in enumerate(zip(blocks, coords)):
        print(f"\n--- Entrenando bloque {idx+1}/{len(blocks)} at ({i},{j}) ---")
        p_enc, p_dec, hist = train_block(block_flat, verbose=True)
        all_histories.append(hist)

        # reconstruir con parámetros entrenados
        amps, norm = normalize_amplitudes(block_flat)
        latent = np.array(qnode_encoder(amps, p_enc), requires_grad=False)
        probs = qnode_decoder(latent, p_dec)

        # recon amplitudes (aprox)
        amps_recon = np.sqrt(np.clip(probs, 0, 1)) * (norm if norm != 0 else 1.0)
        block_recon_2d = amps_recon.reshape((BLOCK_SIZE, BLOCK_SIZE), order='F')
        reconstructed[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = np.clip(block_recon_2d, 0, 255)

        # compressed image 2x2 for visualization
        # map latent [-1,1] -> [0,255]
        lat_pixels = ((latent + 1) * 127.5).astype(np.uint8)
        compressed_small[i//2:(i//2)+2, j//2:(j//2)+2] = lat_pixels.reshape((2,2))

        print(f" Block trained. Final MSE (last) = {hist['mse'][-1]:.6f}, SSIM≈{hist['ssim'][-1]:.4f}")

    t1 = time.time()
    print(f"\nTotal training time: {t1-t0:.1f} s")

    # Guardar imágenes
    Image.fromarray(np.clip(reconstructed, 0, 255).astype(np.uint8)).save(os.path.join(OUT_DIR, "reconstructed.png"))
    Image.fromarray(np.clip(compressed_small, 0, 255).astype(np.uint8)).save(os.path.join(OUT_DIR, "compressed_small.png"))
    Image.fromarray(img_array.astype(np.uint8)).save(os.path.join(OUT_DIR, "original_resized.png"))

    # Métricas globales aproximadas
    # calculamos MSE y SSIM entre original y reconstruida (imagen completa)
    mse_global = np.mean((img_array - reconstructed)**2)
    # SSIM requiere uint8
    orig_u8 = img_array.astype(np.uint8)
    recon_u8 = np.clip(reconstructed, 0, 255).astype(np.uint8)
    try:
        ssim_global = ssim(orig_u8, recon_u8, data_range=255)
    except Exception:
        ssim_global = float("nan")

    print(f"\nMSE global: {mse_global:.6f}")
    print(f"SSIM global: {ssim_global}")

    if visualize:
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.imshow(orig_u8, cmap="gray"); plt.title("Original"); plt.axis("off")
        plt.subplot(1,3,2); plt.imshow(compressed_small.astype(np.uint8), cmap="gray"); plt.title("Compressed (small)"); plt.axis("off")
        plt.subplot(1,3,3); plt.imshow(recon_u8, cmap="gray"); plt.title("Reconstructed"); plt.axis("off")
        plt.tight_layout()
        plt.show()

    return {
        "reconstructed": reconstructed,
        "compressed_small": compressed_small,
        "mse_global": mse_global,
        "ssim_global": ssim_global,
        "histories": all_histories
    }

# ---------------------------
#  MAIN
# ---------------------------
if __name__ == "__main__":
    start = time.time()
    print("Iniciando pipeline de entrenamiento cuántico (por bloques)...")
    results = process_image_train(IMAGE_PATH, visualize=True)
    end = time.time()
    print(f"Tiempo total script: {end-start:.1f} s")
