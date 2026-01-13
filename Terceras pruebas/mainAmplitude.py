import pennylane as qml
from pennylane import numpy as np
from PIL import Image
import time
import glob
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import inicializa_params
import funciones_estado


# ------------------ Configuración variables iniciales ------------------


#DATOS SATELITALES
dataset = "Artificiales experimentos" #Es para el log
num_de_imagenes = 50 #Es para el log

# 0/*: Artificiales
# 1/*: Naturales
# */*: Todas (Artificiales y Naturales)
files = glob.glob("SAR_Dataset/0/4.png")



# DATOS MNIST
files = glob.glob("C:\\Users\\jbelio\\.cache\\kagglehub\\datasets\\ben519\\mnist-as-png\\versions\\1\\mnist-png\\train\\0\\train_image_1.png")



img_save_path = "Resultados/original_resized.png" # Ruta para guardar la imagen redimensionada en base a la original
compressed_save_path = "Resultados/compressed_output.png" # Ruta para guardar la imagen comprimida
reconstructed_save_path = "Resultados/reconstructed_output.png" # Ruta para guardar la imagen comprimida
ruta_log = "Resultados/log.txt" # Ruta del archivo de log con resultados

tasa_de_aprendizaje = 0.05 # Tasa de aprendizaje para el optimizador

resize_dim = (28, 28) # Dimensiones para redimensionar la imagen original
compressed_dim = (14, 14) # Dimensiones de la imagen comprimida
block_size = 4 # Tamaño de bloque para dividir la imagen, por ejemplo, bloques de 4x4 píxeles
# Número de bloques por lado
num_blocks_row = resize_dim[0] // block_size   
num_blocks_col = resize_dim[1] // block_size   
output_block_size = 2   # Porque tiene 4 Z-vals → 2×2 image
compressed_height = num_blocks_row * output_block_size   
compressed_width  = num_blocks_col * output_block_size   
n_qubits = 4 # Número de qubits para representar cada bloque, tiene que ser consistente con el tamaño del bloque (2^n_qubits = block_size*block_size)

log_ratios = [] # Lista para almacenar los ratios de compresión de cada imagen procesada
log_tamaño_original = [] # Lista para almacenar los tamaños originales de las imágenes
log_tamaño_comprimido = [] # Lista para almacenar los tamaños comprimidos de las imágenes

num_iteraciones_global = 5 # Número de iteraciones globales para el entrenamiento, es decir, cuántas veces se optimizan todos los bloques de la imagen
num_iteraciones_bloque = 10  # Número de iteraciones locales para optimizar cada bloque individualmente


optimizer_name = "Adam" # Nombre del optimizador a usar
if optimizer_name == "Adam":
    opt_enc = qml.AdamOptimizer(stepsize=tasa_de_aprendizaje)
    opt_dec = qml.AdamOptimizer(stepsize=tasa_de_aprendizaje)
elif optimizer_name == "GradientDescent":
    opt_enc = qml.GradientDescentOptimizer(stepsize=tasa_de_aprendizaje)
    opt_dec = qml.GradientDescentOptimizer(stepsize=tasa_de_aprendizaje)


tecnica_de_encoding_ansatz = {
    "Amplitude": 1,
    "RotacionesY": 1,
    "CNOT": 1
} # Técnica de encoding a usar en el circuito cuántico, puede ser una lista de técnicas para aplicar secuencialmente

tecnica_de_decoding_ansatz = {
    "Angle": 1,
    "RotacionesY": 1,
    "CNOT": 1
} # Técnica de decoding a usar en el circuito cuántico.
# ¡¡IMPORTANTE!! Este codigo utilizad de embedding amplitude, el cual hacer la compresion. 
# Luego por ello el decoder no va a ser el inverso del decoder para nada.


#num_layers = 2 # Número de capas para los parámetros del encoder y decoder

# CHECKEAR EL ENTRENAMIENTO DEL ROTACIONAL EN EL DECODER
mse_por_bloque = [] # Lista para almacenar los MSE por bloque durante la reconstrucción
ssim_por_bloque = []



start_time = time.time() # Tiempo de inicio para medir el tiempo total de ejecución

n_qubits_dec = n_qubits

#block_size * block_size  # 4x4 = 16 qubits

dev = qml.device("default.qubit", wires=n_qubits) # Dispositivo cuántico simulado (el de por defecto)
dev_dec = qml.device("default.qubit", wires=n_qubits_dec) # Dispositivo cuántico simulado para el decoder

# ------------------ Configuración variables iniciales ------------------

# ------------------ Inicializar parámetros por bloque ------------------


params_dec = {}
params_por_bloque = {}

params_dec, params_por_bloque = inicializa_params.inic_params(block_size, resize_dim, n_qubits_dec, n_qubits)


# ------------------ Inicializar parámetros por bloque ------------------
                                                                                
# ------------------ Entrenamiento global ------------------

# --- Métricas de entrenamiento ---
train_mse_history = []      # MSE medio por iteración
train_ssim_history = []     # SSIM medio por iteración
train_iter_history = []     # Índice global de iteración
global_iter = 0

alpha = 1.0  # Peso para el MSE en la función de pérdida combinada
betta = 0.0  # Peso para el SSIM en la función de pérdida


compressed_img_small = np.zeros((compressed_height, compressed_width)) # Imagen comprimida inicializada en ceros
reconstructed_img_small = np.zeros((resize_dim[0], resize_dim[1])) # Imagen reconstruida inicializada en ceros
idx = 0
num_imagen = 0
for f in files:
    num_imagen += 1
    print(f"Reconstruyendo imagen comprimida {num_imagen}/{len(files)}...")
    reconstructed_blocks = []
    # img = Image.open(f).resize(resize_dim) # Redimensionar
    img = Image.open(f)
    img_array = np.array(img)

    print("\n=== ENTRENAMIENTO AUTOENCODER ===")

    for epoch in range(num_iteraciones_global):
        print(f"\nEpoch {epoch+1}/{num_iteraciones_global}")
        bloque_num = 0
        for i in range(0, resize_dim[0], block_size):
            for j in range(0, resize_dim[1], block_size):
                bloque_num+=1
                print(f"\nBloque {bloque_num}/{(resize_dim[0]*resize_dim[1])/(block_size*block_size)}")

                state, block_norm, block_sum = funciones_estado.imagen_flatten(
                    img_array, i, j, block_size
                )

                print("Block sum:", block_sum)

                params_enc = params_por_bloque[(i, j)]
                params_dec_block = params_dec[(i, j)]

                for iter in range(num_iteraciones_bloque):

                    # ---- ENTRENAMIENTO ----
                    params_enc, params_dec_block = funciones_estado.optimizar_autoencoder_bloque(
                        alpha, betta, dev, dev_dec, n_qubits, opt_enc, opt_dec, params_enc, params_dec_block, state, block_norm, tecnica_de_encoding_ansatz, tecnica_de_decoding_ansatz, block_sum
                    )

                    # ---- MEDICIÓN ----
                    z_vals = funciones_estado.medicion(dev, n_qubits, state, params_enc, tecnica_de_encoding_ansatz, block_sum)
                    z_vals_decoder = funciones_estado.medicion_decoder(dev_dec, n_qubits, z_vals, params_dec_block, tecnica_de_decoding_ansatz, block_sum)


                    # ---- EVALUACIÓN (pipeline completo) ----
                    mse = funciones_estado.loss_autoencoder_block(alpha, betta, block_norm, z_vals_decoder)

                    ssim_val = funciones_estado.ssim_autoencoder_block(block_norm, z_vals_decoder)
                    print(f" Iteración {iter+1}/{num_iteraciones_bloque} - MSE: {mse:.6f}, SSIM: {ssim_val:.6f}")
                    
                    train_mse_history.append(mse)
                    train_ssim_history.append(ssim_val)
                    train_iter_history.append(global_iter)

                    global_iter += 1

                    if iter == 3 and mse < 1e-6:
                        print("MSE muy bajo, saliendo del entrenamiento local del bloque.")
                        break

                params_por_bloque[(i, j)] = params_enc
                params_dec[(i, j)] = params_dec_block


    print("\n=== RECONSTRUCCIÓN ===")

    for i in range(0, resize_dim[0], block_size): # Iterar sobre la imagen en pasos del tamaño del bloque (filas)
        for j in range(0, resize_dim[1], block_size): # Iterar sobre la imagen en pasos del tamaño del bloque (columnas)
            state, block_norm, block_sum = funciones_estado.imagen_flatten(img_array, i, j, block_size)


            params_enc = params_por_bloque[(i, j)]  
            params_dec_block = params_dec[(i, j)]

            z_vals = funciones_estado.medicion(dev, n_qubits, state, params_enc, tecnica_de_encoding_ansatz, block_sum)
            
            compressed_img_small[i//2:(i//2)+2,
                                j//2:(j//2)+2] = funciones_estado.escalar_generar_imagen_mediciones_encoder(z_vals, block_sum)
            
            
            # Decoder 

            z_vals_decoder = funciones_estado.medicion_decoder(dev_dec, n_qubits, z_vals, params_dec_block, tecnica_de_decoding_ansatz, block_sum)
            
            reconstructed_img_small[i:(i+block_size), 
                                    j:(j+block_size)] = funciones_estado.escalar_generar_imagen_mediciones_decoder(z_vals_decoder)


            mse = funciones_estado.mse_autoencoder_block(
                        block_norm, z_vals_decoder
                    )
            ssim_val = funciones_estado.ssim_autoencoder_block(block_norm, z_vals_decoder)
            
            print(f"\nMSE: {mse:.6f}, SSIM: {ssim_val:.6f}")


            mse_por_bloque.append(mse)
            ssim_por_bloque.append(ssim_val)
                
            idx += 1

    # Guardar las imágenes resultantes

    Image.fromarray(compressed_img_small.astype(np.uint8)).save(compressed_save_path)
    Image.fromarray(img_array.astype(np.uint8)).save(img_save_path) # Guardar la imagen redimensionada original
    Image.fromarray(reconstructed_img_small.astype(np.uint8)).save(reconstructed_save_path)

    # funciones_estado.graficar(
    #     img_array, resize_dim,
    #     compressed_img_small, compressed_dim,
    #     reconstructed_img_small,
    #     show_values=True
    # )

    funciones_estado.graficar(img_array, resize_dim, compressed_img_small, compressed_dim, reconstructed_img_small)


    inicial_original_disk_size = funciones_estado.obtener_tamaño(f)
    original_disk_size = funciones_estado.obtener_tamaño(img_save_path)
    compressed_disk_size = funciones_estado.obtener_tamaño(compressed_save_path)
    disk_ratio = original_disk_size / compressed_disk_size if compressed_disk_size > 0 else float('inf')

    # Guardar ratio de compresión de cada imagen
    log_tamaño_original.append(original_disk_size)
    log_tamaño_comprimido.append(compressed_disk_size)
    log_ratios.append(disk_ratio)
    print(f"Imagen {num_imagen}: Tamano original = {original_disk_size:.6f} MB, Tamaño comprimido = {compressed_disk_size:.6f} MB, Ratio de compresion = {disk_ratio:.2f}x")
    
# ------------------ Métricas ------------------

#Media de MSE global
media_MSE = np.mean(mse_por_bloque)
media_SSIM = np.mean(ssim_por_bloque)

# Media de ratios de compresión

average_disk_ratio = np.mean(log_ratios) if log_ratios else 0
average_tamaño_original = np.mean(log_tamaño_original) if log_tamaño_original else 0
average_tamaño_comprimido = np.mean(log_tamaño_comprimido) if log_tamaño_comprimido else 0



end_time = time.time() # Tiempo de fin para medir el tiempo total de ejecución
tiempo_total = end_time - start_time
print(f"Tiempo total: {tiempo_total:.2f} segundos")

funciones_estado.guardar_log(ruta_log, 
                             resize_dim, 
                             compressed_dim, 
                             block_size, 
                             n_qubits, 
                             tecnica_de_encoding_ansatz, 
                             dataset, 
                             num_de_imagenes, 
                             tiempo_total, 
                             average_disk_ratio, 
                             average_tamaño_original, 
                             average_tamaño_comprimido)


with open (ruta_log, "a") as f:
    f.write(f"\nMSE medio global de todas las imagenes: {media_MSE:.6f}\n")
    f.write(f"\nSSIM medio global de todas las imagenes: {media_SSIM:.6f}\n")



funciones_estado.graficar_MSE_SSIM(train_iter_history, train_mse_history, train_ssim_history)



# --- MSE --- 
img1 = np.array(Image.open(img_save_path)).astype(float)
img2 = np.array(Image.open(reconstructed_save_path)).astype(float)

mse_value = np.mean((img1 - img2) ** 2)


# --- SSIM ---
# Para imágenes pequeñas, define win_size <= tamaño más pequeño
ssim_value = ssim(img1, img2, data_range=img1.max() - img1.min())


print(f"MSE prueba 1: {mse_value}")
print (f"MSE prueba 2: {qml.numpy.mean((img1 - img2) ** 2)}")
print(f"SSIM prueba: {ssim_value}")



# ESTO LO GUARDO POR SI EN UN FUTURO PUEDO PONER EN EL TFM QUE HACER EL REDIMENSIONAMIENTO NO 
# TIENE SENTIDO EN COMPARACIÓN CON HACER LA RECONTSUCCION, PUEDO ENSEÑAR LAS METRICAS Y VER QUE SON MALISIMAS

#################CALCULO SSIM (REDIMENSIONAR COMPRIMIDA A ORIGINAL)#################

orig = np.array(Image.open(img_save_path))      
comp = np.array(Image.open(compressed_save_path)) 

comp_resized = resize(comp, orig.shape, order=1)

comp_resized_u8 = (comp_resized * 255).astype(np.uint8) if comp_resized.max() <= 1 else comp_resized.astype(np.uint8)


score = ssim(orig, comp_resized_u8)
#print("SSIM 1 redimensionado comprimida a original = ", score)

#################CALCULO SSIM (REDIMENSIONAR ORIGINA A COMPRIMIDA)#################

orig_small = resize(orig, comp.shape, order=1)
orig_small_u8 = (orig_small * 255).astype(np.uint8)

score = ssim(orig_small_u8, comp.astype(np.uint8))
#print("SSIM 2 redimensionado original a comprimida = ", score)