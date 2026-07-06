import mainAngle_BasisAutoencoder
import mainAmplitude_DenseAutoencoder
import mainSQOriginal


BASIS = "False"
DENSE = "False"
DATASET = "MNIST"

N_TRAIN_POR_NUMERO = 20
N_TEST_POR_NUMERO = 5
DEV_SIZE = 0.2
N_EPOCHS = 20

MEJORA = False
ANSATZ = "StronglyEntangling"
BATCH_SIZE = 16
N_LAYERS_VAL = 6
K_BLOQUES = 5
LR = 0.035

mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(DENSE, DATASET, N_EPOCHS, BATCH_SIZE, N_LAYERS_VAL, N_TRAIN_POR_NUMERO, N_TEST_POR_NUMERO, ANSATZ, MEJORA, K_BLOQUES, DEV_SIZE, LR)
