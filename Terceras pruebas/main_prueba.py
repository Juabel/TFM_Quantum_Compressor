import mainAngle_BasisAutoencoder
import mainAmplitude_DenseAutoencoder
import mainSQOriginal
from itertools import product


# # =========================
# # CONFIGURACIÓN FIJA
# # =========================
# BASIS = "False"
# DENSE = "False"
# DATASET = "MNIST"

# N_TRAIN_POR_NUMERO = 5
# N_TEST_POR_NUMERO = 2
# DEV_SIZE = 0.2
# N_EPOCHS = 100

# MEJORA = False
# ANSATZ = "StronglyEntangling"


# # =========================
# # EXPERIMENTOS
# # =========================
# K_LIST = [20]
# BATCH_LIST = [16]
# LR_LIST = [0.05, 0.01, 0.005]
# LAYERS_LIST = [6]


# def print_config(k, batch, lr, layers):
#     print("\n" + "=" * 70)
#     print("🚀 EJECUCIÓN AUTOENCODER CUÁNTICO")
#     print("=" * 70)
#     print(f"DENSE            : {DENSE}")
#     print(f"DATASET          : {DATASET}")
#     print(f"EPOCHS           : {N_EPOCHS}")
#     print(f"BATCH_SIZE       : {batch}")
#     print(f"N_LAYERS         : {layers}")
#     print(f"K_BLOQUES        : {k}")
#     print(f"LEARNING_RATE    : {lr}")
#     print(f"ANSATZ           : {ANSATZ}")
#     print("=" * 70 + "\n")

# experiments = list(
#     product(
#         K_LIST,
#         BATCH_LIST,
#         LR_LIST,
#         LAYERS_LIST
#     )
# )

# # =========================
# # EJECUCIÓN
# # =========================
# print(f"\nTOTAL EXPERIMENTOS: {len(experiments)}\n")

# for i, (k, batch, lr, layers) in enumerate(experiments):

#     print(f"\n🧪 EXPERIMENTO {i+1}/{len(experiments)}")

#     print_config(k, batch, lr, layers)

#     mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(
#         DENSE,
#         DATASET,
#         N_EPOCHS,
#         batch,
#         layers,
#         N_TRAIN_POR_NUMERO,
#         N_TEST_POR_NUMERO,
#         ANSATZ,
#         MEJORA,
#         k,
#         DEV_SIZE,
#         lr
#     )









# print("Dense Angle \n")
# DENSE = "True"
# mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(DENSE, DATASET, N_EPOCHS, BATCH_SIZE, N_LAYERS_VAL, N_TRAIN_POR_NUMERO, N_TEST_POR_NUMERO, ANSATZ, MEJORA, K_BLOQUES, DEV_SIZE)
# print("Angle \n")
# mainAngle_BasisAutoencoder.ejecutar_autoencoder(BASIS, DATASET, N_EPOCHS, BATCH_SIZE, N_LAYERS_VAL, N_TRAIN_POR_NUMERO, N_TEST_POR_NUMERO, ANSATZ, MEJORA, K_BLOQUES, DEV_SIZE)
# print("Basis \n")
# BASIS = "True"
# mainAngle_BasisAutoencoder.ejecutar_autoencoder(BASIS, DATASET, N_EPOCHS, BATCH_SIZE, N_LAYERS_VAL, N_TRAIN_POR_NUMERO, N_TEST_POR_NUMERO, ANSATZ, MEJORA, K_BLOQUES, DEV_SIZE)
# print("Single Qubit \n")
# mainSQOriginal.ejecutar_autoencoder(DATASET, N_EPOCHS, BATCH_SIZE, N_LAYERS_VAL, N_TRAIN_POR_NUMERO, N_TEST_POR_NUMERO, ANSATZ, MEJORA, K_BLOQUES, DEV_SIZE)



BASIS = "False"
DENSE = "False"
DATASET = "SAR" # Es para el log
N_TRAIN_POR_NUMERO = 10
N_TEST_POR_NUMERO = 1
DEV_SIZE = 0.2
N_EPOCHS = 3
BATCH_SIZE = 4
N_LAYERS_VAL = 6
MEJORA = False # Si se quiere usar la mejora propuesta en el encoder (solo para el ansatz "Mejora")
ANSATZ = "StronglyEntangling" # Nombre del ansatz a usar en el encoder y decoder ("StronglyEntangling", "Paper")
K_BLOQUES = 5
LR = 0.2


mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(DENSE, DATASET, N_EPOCHS, BATCH_SIZE, N_LAYERS_VAL, N_TRAIN_POR_NUMERO, N_TEST_POR_NUMERO, ANSATZ, MEJORA, K_BLOQUES, DEV_SIZE, LR)
# DENSE = "True"
# mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(DENSE, DATASET, N_EPOCHS, BATCH_SIZE, N_LAYERS_VAL, N_TRAIN_POR_NUMERO, N_TEST_POR_NUMERO, ANSATZ, MEJORA, K_BLOQUES, DEV_SIZE, LR)
# print("Angle \n")
# mainAngle_BasisAutoencoder.ejecutar_autoencoder(BASIS, DATASET, N_EPOCHS, BATCH_SIZE, N_LAYERS_VAL, N_TRAIN_POR_NUMERO, N_TEST_POR_NUMERO, ANSATZ, MEJORA, K_BLOQUES, DEV_SIZE, LR)
# BASIS = "True"
# print("Basis \n")
# mainAngle_BasisAutoencoder.ejecutar_autoencoder(BASIS, DATASET, N_EPOCHS, BATCH_SIZE, N_LAYERS_VAL, N_TRAIN_POR_NUMERO, N_TEST_POR_NUMERO, ANSATZ, MEJORA, K_BLOQUES, DEV_SIZE, LR)
# print("SQ \n")
# mainSQOriginal.ejecutar_autoencoder(DATASET, N_EPOCHS, BATCH_SIZE, N_LAYERS_VAL, N_TRAIN_POR_NUMERO, N_TEST_POR_NUMERO, ANSATZ, MEJORA, K_BLOQUES, DEV_SIZE, LR)