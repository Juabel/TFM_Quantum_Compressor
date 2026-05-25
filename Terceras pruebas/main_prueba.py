import mainAngle_BasisAutoencoder
import mainAmplitude_DenseAutoencoder
import mainSQOriginal

basis = "False"
dense = "False"
dataset = "MNIST" #Es para el log
n_train_por_numero = 20
n_test_por_numero = 10
n_epochs = 2
batch_size = 10
n_layers_val = 3
mejora = False # Si se quiere usar la mejora propuesta en el encoder (solo para el ansatz "Mejora")
ansatz = "StronglyEntangling" # Nombre del ansatz a usar en el encoder y decoder ("StronglyEntangling", "Paper")
k_bloques = 20


mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_epochs, batch_size, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora, k_bloques)
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_epochs, batch_size, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora, k_bloques)
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_epochs, batch_size, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora, k_bloques)
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_epochs, batch_size, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora, k_bloques)
mainSQOriginal.ejecutar_autoencoder(dataset, n_epochs, batch_size, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora, k_bloques)



basis = "False"
dense = "False"
dataset = "SAR" #Es para el log
n_train_por_numero = 40
n_test_por_numero = 10
n_epochs = 2
batch_size = 10
n_layers_val = 3
mejora = False # Si se quiere usar la mejora propuesta en el encoder (solo para el ansatz "Mejora")
ansatz = "StronglyEntangling" # Nombre del ansatz a usar en el encoder y decoder ("StronglyEntangling", "Paper")
k_bloques = 100


mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_epochs, batch_size, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora, k_bloques)
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_epochs, batch_size, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora, k_bloques)
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_epochs, batch_size, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora, k_bloques)
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_epochs, batch_size, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora, k_bloques)
mainSQOriginal.ejecutar_autoencoder(dataset, n_epochs, batch_size, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora, k_bloques)

