import mainAngle_BasisAutoencoder
import mainAmplitude_DenseAutoencoder
import mainSQOriginal

basis = "False"
dense = "False"
dataset = "MNIST" # "MNIST" o "SAR"
n_global = 2
n_local = 10
n_layers_val = 5
mejora = False # Si se quiere usar la mejora propuesta en el encoder (solo para el ansatz "Mejora")
n_train_por_numero = 20
n_test_por_numero = 5

ansatz = "StronglyEntangling" # Nombre del ansatz a usar en el encoder y decoder ("StronglyEntangling", "Paper")

print("\n MNIST CON 200 IMAGENES, STRONGLY, 5 LAYERS, 10 LOCAL\n")

print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n MNIST CON 200 IMAGENES, STRONGLY, 10 LAYERS, 10 LOCAL\n")


n_layers_val = 10

print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n MNIST CON 200 IMAGENES, STRONGLY, 15 LAYERS, 10 LOCAL\n")


n_layers_val = 15

print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n MNIST CON 200 IMAGENES, PAPER MEJORA, 5 LAYERS, 10 LOCAL\n")


n_layers_val = 5
mejora = True # Si se quiere usar la mejora propuesta en el encoder (solo para el ansatz "Mejora")
ansatz = "Paper" # Nombre del ansatz a usar en el encoder y decoder ("StronglyEntangling", "Paper")



print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n MNIST CON 200 IMAGENES, PAPER MEJORA, 10 LAYERS, 10 LOCAL\n")


n_layers_val = 10


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n MNIST CON 200 IMAGENES, PAPER MEJORA, 15 LAYERS, 10 LOCAL\n")


n_layers_val = 15


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)




print("\n MNIST CON 200 IMAGENES, PAPER SIN MEJORA, 5 LAYERS, 10 LOCAL\n")


n_layers_val = 5
mejora = False # Si se quiere usar la mejora propuesta en el encoder (solo para el ansatz "Mejora")
ansatz = "Paper" # Nombre del ansatz a usar en el encoder y decoder ("StronglyEntangling", "Paper")



print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n MNIST CON 200 IMAGENES, PAPER SIN MEJORA, 10 LAYERS, 10 LOCAL\n")


n_layers_val = 10


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n MNIST CON 200 IMAGENES, PAPER SIN MEJORA, 15 LAYERS, 10 LOCAL\n")


n_layers_val = 15


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)

















dataset = "SAR" 
ansatz = "StronglyEntangling" # Nombre del ansatz a usar en el encoder y decoder ("StronglyEntangling", "Paper")
n_layers_val = 5

print("\n SAR CON 80 IMAGENES, STRONGLY, 5 LAYERS, 10 LOCAL\n")

print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 80 imagenes, 5 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 80 imagenes, 5 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 80 imagenes, 5 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 80 imagenes, 5 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 80 imagenes, 5 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)

print("\n SAR CON 80 IMAGENES, STRONGLY, 10 LAYERS, 10 LOCAL\n")


n_layers_val = 10

print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)



print("\n SAR CON 80 IMAGENES, STRONGLY, 15 LAYERS, 10 LOCAL\n")


n_layers_val = 15

print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)




print("\n SAR CON 80 IMAGENES, PAPER MEJORA, 5 LAYERS, 10 LOCAL\n")


n_layers_val = 5
mejora = True # Si se quiere usar la mejora propuesta en el encoder (solo para el ansatz "Mejora")
ansatz = "Paper" # Nombre del ansatz a usar en el encoder y decoder ("StronglyEntangling", "Paper")


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 80 imagenes, 5 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 80 imagenes, 5 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 80 imagenes, 5 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 80 imagenes, 5 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 80 imagenes, 5 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n SAR CON 80 IMAGENES, PAPER MEJORA, 10 LAYERS, 10 LOCAL\n")


n_layers_val = 10

print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)

print("\n SAR CON 80 IMAGENES, PAPER MEJORA, 15 LAYERS, 10 LOCAL\n")


n_layers_val = 15


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)




print("\n SAR CON 200 IMAGENES, PAPER SIN MEJORA, 5 LAYERS, 10 LOCAL\n")


n_layers_val = 5
mejora = False # Si se quiere usar la mejora propuesta en el encoder (solo para el ansatz "Mejora")
ansatz = "Paper" # Nombre del ansatz a usar en el encoder y decoder ("StronglyEntangling", "Paper")



print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 5 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n SAR CON 200 IMAGENES, PAPER SIN MEJORA, 10 LAYERS, 10 LOCAL\n")


n_layers_val = 10


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n SAR CON 200 IMAGENES, PAPER SIN MEJORA, 15 LAYERS, 10 LOCAL\n")


n_layers_val = 15


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)



basis = "False"
dense = "False"
dataset = "MNIST" # "MNIST" o "SAR"
n_global = 2
n_local = 15
n_layers_val = 5
mejora = False # Si se quiere usar la mejora propuesta en el encoder (solo para el ansatz "Mejora")
n_train_por_numero = 20
n_test_por_numero = 5


ansatz = "StronglyEntangling" # Nombre del ansatz a usar en el encoder y decoder ("StronglyEntangling", "Paper")

print("\n MNIST CON 200 IMAGENES, STRONGLY, 5 LAYERS, 15 LOCAL \n")

print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n MNIST CON 200 IMAGENES, STRONGLY, 10 LAYERS, 15 LOCAL\n")


n_layers_val = 10

print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n MNIST CON 200 IMAGENES, STRONGLY, 15 LAYERS, 15 LOCAL\n")


n_layers_val = 15

print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n MNIST CON 200 IMAGENES, PAPER MEJORA, 5 LAYERS, 15 LOCAL\n")


n_layers_val = 5
mejora = True # Si se quiere usar la mejora propuesta en el encoder (solo para el ansatz "Mejora")
ansatz = "Paper" # Nombre del ansatz a usar en el encoder y decoder ("StronglyEntangling", "Paper")



print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n MNIST CON 200 IMAGENES, PAPER MEJORA, 10 LAYERS, 15 LOCAL\n")


n_layers_val = 10


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n MNIST CON 200 IMAGENES, PAPER MEJORA, 15 LAYERS, 15 LOCAL\n")


n_layers_val = 15


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)




print("\n MNIST CON 200 IMAGENES, PAPER SIN MEJORA, 5 LAYERS, 15 LOCAL\n")


n_layers_val = 5
mejora = False # Si se quiere usar la mejora propuesta en el encoder (solo para el ansatz "Mejora")
ansatz = "Paper" # Nombre del ansatz a usar en el encoder y decoder ("StronglyEntangling", "Paper")



print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n MNIST CON 200 IMAGENES, PAPER SIN MEJORA, 10 LAYERS, 10 LOCAL\n")


n_layers_val = 10


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n MNIST CON 200 IMAGENES, PAPER SIN MEJORA, 15 LAYERS, 10 LOCAL\n")


n_layers_val = 15


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)

















dataset = "SAR" 
ansatz = "StronglyEntangling" # Nombre del ansatz a usar en el encoder y decoder ("StronglyEntangling", "Paper")
n_layers_val = 5

print("\n SAR CON 80 IMAGENES, STRONGLY, 5 LAYERS, 15 LOCAL\n")

print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 80 imagenes, 5 layers, 15 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 80 imagenes, 5 layers, 15 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 80 imagenes, 5 layers, 15 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 80 imagenes, 5 layers, 15 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 80 imagenes, 5 layers, 15 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)

print("\n SAR CON 80 IMAGENES, STRONGLY, 10 LAYERS, 15 LOCAL\n")


n_layers_val = 10

print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 80 imagenes, 10 layers, 15 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 80 imagenes, 10 layers, 15 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 80 imagenes, 10 layers, 15 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 80 imagenes, 10 layers, 15 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 80 imagenes, 10 layers, 15 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)



print("\n SAR CON 80 IMAGENES, STRONGLY, 15 LAYERS, 15 LOCAL\n")


n_layers_val = 15

print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 80 imagenes, 15 layers, 15 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 80 imagenes, 15 layers, 15 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 80 imagenes, 15 layers, 15 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 80 imagenes, 15 layers, 15 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 80 imagenes, 15 layers, 15 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)




print("\n SAR CON 80 IMAGENES, PAPER MEJORA, 5 LAYERS, 15 LOCAL\n")


n_layers_val = 5
mejora = True # Si se quiere usar la mejora propuesta en el encoder (solo para el ansatz "Mejora")
ansatz = "Paper" # Nombre del ansatz a usar en el encoder y decoder ("StronglyEntangling", "Paper")


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 80 imagenes, 5 layers, 15 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 80 imagenes, 5 layers, 15 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 80 imagenes, 5 layers, 15 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 80 imagenes, 5 layers, 15 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 80 imagenes, 5 layers, 15 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n SAR CON 80 IMAGENES, PAPER MEJORA, 10 LAYERS, 15 LOCAL\n")


n_layers_val = 10

print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 80 imagenes, 10 layers, 15 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 80 imagenes, 10 layers, 15 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 80 imagenes, 10 layers, 15 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 80 imagenes, 10 layers, 15 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 80 imagenes, 10 layers, 15 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)

print("\n SAR CON 80 IMAGENES, PAPER MEJORA, 15 LAYERS, 15 LOCAL\n")


n_layers_val = 15


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 80 imagenes, 15 layers, 15 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 80 imagenes, 15 layers, 15 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 80 imagenes, 15 layers, 15 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 80 imagenes, 15 layers, 15 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 80 imagenes, 15 layers, 15 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)




print("\n SAR CON 200 IMAGENES, PAPER SIN MEJORA, 5 LAYERS, 15 LOCAL\n")


n_layers_val = 5
mejora = False # Si se quiere usar la mejora propuesta en el encoder (solo para el ansatz "Mejora")
ansatz = "Paper" # Nombre del ansatz a usar en el encoder y decoder ("StronglyEntangling", "Paper")



print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 5 layers, 15 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n SAR CON 200 IMAGENES, PAPER SIN MEJORA, 10 LAYERS, 10 LOCAL\n")


n_layers_val = 10


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 10 layers, 15 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)


print("\n SAR CON 200 IMAGENES, PAPER SIN MEJORA, 15 LAYERS, 10 LOCAL\n")


n_layers_val = 15


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 15 layers, 15 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero)
