import pennylane as qml
import numpy as np
from PIL import Image
import time
import glob
import inicializa_params
import funciones_estado
import torch
from skimage.metrics import structural_similarity as ssim



# ------------------ Configuración variables iniciales ------------------


#DATOS SATELITALES
dataset = "Artificiales experimentos" #Es para el log
num_de_imagenes = 50 #Es para el log

# 0/*: Artificiales
# 1/*: Naturales
# */*: Todas (Artificiales y Naturales)
# files = glob.glob("SAR_Dataset/0/4.png")



# DATOS MNIST
files = glob.glob("C:\\Users\\jbelio\\.cache\\kagglehub\\datasets\\ben519\\mnist-as-png\\versions\\1\\mnist-png\\train\\0\\train_image_1.png")



img_save_path = "Resultados/original_resized.png" # Ruta para guardar la imagen redimensionada en base a la original
compressed_save_path = "Resultados/compressed_output.png" # Ruta para guardar la imagen comprimida
reconstructed_save_path = "Resultados/reconstructed_output.png" # Ruta para guardar la imagen comprimida
ruta_log = "Resultados/log.txt" # Ruta del archivo de log con resultados

tasa_de_aprendizaje = 0.05 # Tasa de aprendizaje para el optimizador

output_block_size_height = 2   # Porque tiene 4 Z-vals → 2×2 image
output_block_size_width = 2 # Porque tiene 4 Z-vals → 2×2 image



# resize_dim = (400, 400) # Dimensiones para redimensionar la imagen original

resize_dim = (28, 28) # Dimensiones para redimensionar la imagen original

block_size = 4 # Tamaño de bloque para dividir la imagen, por ejemplo, bloques de 4x4 píxeles
n_blocks_h = resize_dim[0] // block_size
n_blocks_w = resize_dim[1] // block_size


compressed_dim = (
    n_blocks_h * output_block_size_height,
    n_blocks_w * output_block_size_width
)


# Dimesiones comprimida
compressed_rows = compressed_dim[0]   
compressed_cols = compressed_dim[1]

n_qubits = 4 # Número de qubits para representar cada bloque, tiene que ser consistente con el tamaño del bloque (2^n_qubits = block_size*block_size)


#DEFINIR ESTO

#SABER SI VA EN COLUMNAS O FILAS


log_ratios = [] # Lista para almacenar los ratios de compresión de cada imagen procesada
log_tamaño_original = [] # Lista para almacenar los tamaños originales de las imágenes
log_tamaño_comprimido = [] # Lista para almacenar los tamaños comprimidos de las imágenes

num_iteraciones_global = 2 # Número de iteraciones globales para el entrenamiento, es decir, cuántas veces se optimizan todos los bloques de la imagen
num_iteraciones_bloque = 20 # Número de iteraciones locales para optimizar cada bloque individualmente


optimizer_name = "Adam" # Nombre del optimizador a usar


# if optimizer_name == "Adam":
#     opt = qml.AdamOptimizer(stepsize=tasa_de_aprendizaje)
# elif optimizer_name == "GradientDescent":
#     opt= qml.GradientDescentOptimizer(stepsize=tasa_de_aprendizaje)


tecnica_de_encoding_ansatz = {
    "Amplitude": 1,
    # "RotacionesY": 1,
    # "CNOT": 1
} # Técnica de encoding a usar en el circuito cuántico, puede ser una lista de técnicas para aplicar secuencialmente

tecnica_de_decoding_ansatz = {
    "Angle": 1,
    # "RotacionesY": 1,
    # "CNOT": 1
} # Técnica de decoding a usar en el circuito cuántico.
# ¡¡IMPORTANTE!! Este codigo utilizad de embedding amplitude, el cual hacer la compresion. 
# Luego por ello el decoder no va a ser el inverso del decoder para nada.

entrenamiento = False # Si se quiere entrenar el autoencoder o solo hacer la reconstrucción con parámetros ya entrenados


#num_layers = 2 # Número de capas para los parámetros del encoder y decoder

# CHECKEAR EL ENTRENAMIENTO DEL ROTACIONAL EN EL DECODER
mse_por_bloque = [] # Lista para almacenar los MSE por bloque durante la reconstrucción
ssim_por_bloque = []



start_time = time.time() # Tiempo de inicio para medir el tiempo total de ejecución

n_qubits_dec = n_qubits

#block_size * block_size  # 4x4 = 16 qubits

dev = qml.device("default.qubit", wires=n_qubits) # Dispositivo cuántico simulado (el de por defecto)
dev_dec = qml.device("default.qubit", wires=n_qubits_dec) # Dispositivo cuántico simulado para el decoder
device = "cpu" # Dispositivo para PyTorch (CPU o GPU)

# ------------------ Configuración variables iniciales ------------------

# ------------------ Inicializar parámetros por bloque ------------------




params = inicializa_params.inic_params(block_size, resize_dim, n_qubits_dec, n_qubits)

#Inicializar circuitos (no se usan los circuitos dev y dev_dec, pero si las funciones qnode que crean)
circuit_enc, circuit_dec = funciones_estado.inicializar_circuitos(dev, dev_dec, n_qubits, tecnica_de_encoding_ansatz, tecnica_de_decoding_ansatz)


opt = inicializa_params.crear_optimizador(optimizer_name, params, tasa_de_aprendizaje)


# ------------------ Inicializar parámetros por bloque ------------------
                                                                                
# ------------------ Entrenamiento global ------------------

# --- Métricas de entrenamiento ---
train_mse_history = []      # MSE medio por iteración
train_iter_history = []     # Índice global de iteración
global_iter = 0

alpha = 1.0  # Peso para el MSE en la función de pérdida combinada
betta = 0.0  # Peso para el SSIM en la función de pérdida

compressed_img_small = np.zeros((compressed_rows, compressed_cols))
reconstructed_img_small = np.zeros((resize_dim[0], resize_dim[1]))

idx = 0
num_imagen = 0
for f in files:
    num_imagen += 1
    print(f"Reconstruyendo imagen comprimida {num_imagen}/{len(files)}...")
    reconstructed_blocks = []
    # img = Image.open(f).resize(resize_dim) # Redimensionar
    img = Image.open(f)
    # Redimensionar la imagen a las dimensiones especificadas
    img = img.resize(resize_dim)
    img_array = torch.from_numpy(np.array(img)).float().to(device)


    # Generar listas para guardar mse e iteraciones locales por imagen, de forma que luego por cada iteración global tengo una gráfica
    # Para ello quiero hacer listas de estos valores por indices, de forma que por cada iteracion global tengo una grafica donde sus valores estan guardados en las listas con los indices


    if entrenamiento == True:
        print("\n=== ENTRENAMIENTO AUTOENCODER ===")

        for epoch in range(num_iteraciones_global):
            print(f"\nEpoch {epoch+1}/{num_iteraciones_global}")

            mse_epoch = []
            iter_epoch = []

            bloque_num = 0
            for i in range(0, resize_dim[0], block_size):
                for j in range(0, resize_dim[1], block_size):
                    bloque_num+=1
                    print(f"\nBloque {bloque_num}/{(resize_dim[0]*resize_dim[1])/(block_size*block_size)}")

                    state, block_norm, block_sum = funciones_estado.imagen_flatten(
                        img_array, i, j, block_size, device
                    )
                    print("Estado inicial a utilizar en bloque :", state)

                    if(block_sum.item() == 0.0):
                        print("Bloque de solo ceros, saltando entrenamiento local del bloque.")
                        continue

                    for iter in range(num_iteraciones_bloque):

                        # ---- ENTRENAMIENTO ----
                        params[(i, j)], mse = funciones_estado.optimizar_autoencoder_bloque(
                            alpha, betta, opt, params[(i, j)], state, block_norm, circuit_enc, circuit_dec, block_size
                        )

                        params_enc, params_dec_block = params[(i, j)]

                        print(f" Iteración {iter+1}/{num_iteraciones_bloque} - MSE: {mse:.6f}")

                        if iter == 3 and mse < 1e-6:
                            print("MSE muy bajo, saliendo del entrenamiento local del bloque.")
                            break

                    global_iter += 1
                    if(block_sum.item() == 0.0):
                        mse = 0.0
                    mse_epoch.append(mse)
                    iter_epoch.append(global_iter)
            train_mse_history.append(mse_epoch)
            train_iter_history.append(iter_epoch)



    print("\n=== RECONSTRUCCIÓN ===")

    for i in range(0, resize_dim[0], block_size): # Iterar sobre la imagen en pasos del tamaño del bloque (filas)
        for j in range(0, resize_dim[1], block_size): # Iterar sobre la imagen en pasos del tamaño del bloque (columnas)
            state, block_norm, block_sum = funciones_estado.imagen_flatten(img_array, i, j, block_size, device)

            print("Estado inicial a utilizar en bloque :", state)

            params_enc, params_dec_block = params[(i, j)]

            z_vals = funciones_estado.medicion(state, params_enc, circuit_enc)
            
            # Índices destino en la imagen comprimida
            row_idx = (i // block_size) * output_block_size_height
            col_idx = (j // block_size) * output_block_size_width

            print("Z valores obtenidos por el encoder : ",z_vals)
            print("Block sum (intensidad del bloque original) : ", block_sum)

            output = funciones_estado.escalar_generar_imagen_mediciones_encoder(z_vals, block_sum, output_block_size_height, output_block_size_width)

            print("Output del encoder : ", output)

            compressed_img_small[row_idx:row_idx+output_block_size_height,
                             col_idx:col_idx+output_block_size_width] = output
                        
            # Decoder 

            z_vals_decoder = funciones_estado.medicion_decoder(z_vals, params_dec_block, circuit_dec)

            print("Z valores obtenidos por el decoder : ", z_vals_decoder)

            output_dec = funciones_estado.escalar_generar_imagen_mediciones_decoder(z_vals_decoder, block_sum, block_size)

            print("Output del decoder : ", output_dec)

            reconstructed_img_small[i:(i+block_size), 
                                    j:(j+block_size)] = output_dec

            
            if(block_sum.item() == 0.0):
                mse = 0.0
            else:
                mse = funciones_estado.loss_autoencoder_block(alpha, betta, block_norm, z_vals_decoder, block_size)
            print(f"\nMSE: {mse:.6f}")


            mse_por_bloque.append(mse)
    
            idx += 1

    # Guardar las imágenes resultantes

    Image.fromarray(compressed_img_small.astype(np.uint8)).save(compressed_save_path)
    Image.fromarray(img_array.detach().cpu().numpy().astype(np.uint8)).save(img_save_path)    
    Image.fromarray(reconstructed_img_small.astype(np.uint8)).save(reconstructed_save_path)

    # funciones_estado.graficar(
    #     img_array, resize_dim,
    #     compressed_img_small, compressed_dim,
    #     reconstructed_img_small,
    #     show_values=True
    # )

    funciones_estado.graficar(img_array, resize_dim, compressed_img_small, compressed_dim, reconstructed_img_small)
    funciones_estado.prueba(img_array, resize_dim, compressed_img_small, compressed_dim, reconstructed_img_small, block_size)


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
mse_nonzero = [x for x in mse_por_bloque if x != 0]
media_MSE = sum(mse_nonzero) / len(mse_nonzero) if mse_nonzero else 0.0

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


#Para SSIM
original = img_array.detach().cpu().numpy().astype(np.uint8)
reconstructed = reconstructed_img_small.astype(np.uint8)
ssim_val = ssim(original, reconstructed, data_range=255)



with open (ruta_log, "a") as f:
    f.write(f"\nMSE medio global de todas las imagenes: {media_MSE:.6f}\n")
    f.write(f"SSIM medio de las imagenes: {ssim_val:.6f}\n")


funciones_estado.graficar_MSE(train_iter_history, train_mse_history)

