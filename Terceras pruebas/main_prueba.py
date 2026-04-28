import mainAngle_BasisAutoencoder
import mainAmplitude_DenseAutoencoder
import mainSQOriginal

basis = "False"
dense = "False"
dataset = "SAR" #Es para el log
n_train_por_numero = 1
n_test_por_numero = 1
n_global = 1
n_local = 1
n_layers_val = 2
mejora = False # Si se quiere usar la mejora propuesta en el encoder (solo para el ansatz "Mejora")
ansatz = "StronglyEntangling" # Nombre del ansatz a usar en el encoder y decoder ("StronglyEntangling", "Paper")

mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)