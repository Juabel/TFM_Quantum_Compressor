from traceback import print_tb
import numpy as np
import pennylane as qml
from PIL import Image
import tifffile as tiff
import matplotlib.pyplot as plt



def normalize_image(img):
    img = img.astype(np.float32)
    return img / 255.0

def pad_image(img, block_size):
    h, w = img.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant')

def split_blocks(img, block_size):
    h, w = img.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            blocks.append(img[i:i+block_size, j:j+block_size])
    return blocks

def block_intensity_and_weights(block):
    intensity = np.sum(block)
    if intensity == 0:
        weights = np.zeros_like(block)
    else:
        weights = block / intensity
    return intensity, weights

def intensity_histogram(intensities, B):
    intensities = np.array(intensities)
    max_I = intensities.max() if intensities.max() > 0 else 1.0
    bins = np.linspace(0, max_I, B + 1)
    hist = np.zeros(B, dtype=int)

    for I in intensities:
        idx = np.digitize(I, bins) - 1
        idx = min(idx, B - 1)
        hist[idx] += 1

    return hist, bins

def amplitude_encode(hist):
    hist = hist.astype(np.float32)
    norm = np.linalg.norm(hist)


    if norm == 0:
        raise ValueError("Histograma vacío")
    return hist / norm



def histogram_quantum_state(encoded_hist):
    B = len(encoded_hist)
    n_qubits = int(np.ceil(np.log2(B)))
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        qml.AmplitudeEmbedding(
            features=encoded_hist,
            wires=range(n_qubits),
            normalize=True
        )
        return qml.probs(wires=range(n_qubits))
        #return [qml.expval(qml.PauliZ(k)) for k in range(n_qubits)]

    return circuit()

def compressed_representation(hist_probs):
    return hist_probs


def reconstruct_image(blocks_weights, block_groups, compressed_values, block_size, img_shape):
    h, w = img_shape
    recon = np.zeros((h, w))
    blocks_per_row = w // block_size

    for b, (weights, group) in enumerate(zip(blocks_weights, block_groups)):
        value = compressed_values[group]

        block = value * weights

        i = (b // blocks_per_row) * block_size
        j = (b % blocks_per_row) * block_size
        recon[i:i+block_size, j:j+block_size] = block

    return recon

def quantum_image_compression(img_path, block_size, B):
    
    img = tiff.imread(img_path)

    # Eliminar dimensiones triviales
    img = np.squeeze(img)

    # Si tiene canales, colapsarlos
    if img.ndim == 3:
        img = img.mean(axis=2)

    assert img.ndim == 2, f"Imagen mal formada tras procesar, shape={img.shape}"

    if img.dtype != np.uint8:
        img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255).astype(np.uint8)

            
    # Leer imagen
    #img = Image.open(img_path).convert("L")  # escala de grises
    #img = np.array(img)

    img_norm = normalize_image(img)
    img_pad = pad_image(img_norm, block_size)
    blocks = split_blocks(img_pad, block_size)

    intensities = []
    weights_list = []

    for block in blocks:
        I, W = block_intensity_and_weights(block)
        intensities.append(I)
        weights_list.append(W)

    hist, bins = intensity_histogram(intensities, B)
    #encoded_hist = amplitude_encode(hist)
    encoded_hist = np.sqrt(hist / hist.sum())
    print("Histograma normalizado:", encoded_hist)
    probs = histogram_quantum_state(encoded_hist)
    print("Probabilidades desde estado cuántico:", probs)
    block_groups = [min(np.digitize(I, bins) - 1, B - 1) for I in intensities]

    # Reescala las probabilidades al rango de intensidades
    probs_rescaled = probs * np.mean(intensities) * len(probs)

    recon = reconstruct_image(
        weights_list,
        block_groups,
        probs_rescaled,
        block_size,
        img_pad.shape
    )

    for b in range(B):
        idxs = [i for i, g in enumerate(block_groups) if g == b]
        if idxs:
            print(
                f"Bin {b}:",
                "count =", len(idxs),
                "mean I =", np.mean([intensities[i] for i in idxs]),
                "assigned =", probs_rescaled[b]
            )


    

    # 🔹 IMPORTANTE: devuelve ambas imágenes
    return img_norm, recon[:img.shape[0], :img.shape[1]], hist



if __name__ == "__main__":
    original, reconstructed, hist = quantum_image_compression(
        "jetplane.tif",
        block_size=32,
        B=4
    )
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    plt.bar(range(len(hist)), hist)
    plt.title("Histograma clásico")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original, cmap="gray")   
    plt.axis("off")

    reconstructed_vis = reconstructed.copy()
    reconstructed_vis -= reconstructed_vis.min()
    reconstructed_vis /= reconstructed_vis.max()

    plt.subplot(1, 2, 2)
    plt.title("Reconstruida")
    plt.imshow(reconstructed_vis, cmap="gray")
    plt.axis("off")

    plt.show()