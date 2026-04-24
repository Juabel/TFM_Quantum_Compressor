import mainAngle_BasisAutoencoder
import mainAmplitude_DenseAutoencoder
import mainSQOriginal

basis = "False"
dense = "False"
dataset = "MNIST" #Es para el log
n_train_por_numero = 1
n_test_por_numero = 1
n_global = 1
n_local = 1
n_layers_val = 2
mejora = False # Si se quiere usar la mejora propuesta en el encoder (solo para el ansatz "Mejora")
ansatz = "StronglyEntangling" # Nombre del ansatz a usar en el encoder y decoder ("StronglyEntangling", "Paper")


mejora = False # Si se quiere usar la mejora propuesta en el encoder (solo para el ansatz "Mejora")
ansatz = "Paper" # Nombre del ansatz a usar en el encoder y decoder ("StronglyEntangling", "Paper")
print("\n 6 ")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n 7 ")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n 8 ")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
basis = "True"
dense = "True"
print("\n 9 ")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n 10 ")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora) 
basis = "False"
dense = "False"
