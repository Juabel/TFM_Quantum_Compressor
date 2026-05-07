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

mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)



#------------PRUEBAS QUE NO ME DAN PARA HACER#------------
##------------COMENTAR TODO A PARTIR DE AQUI#------------


print("\n MNIST CON 200 IMAGENES, STRONGLY, 10 LAYERS, 10 LOCAL\n")


n_layers_val = 10

print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
basis = "False"
dense = "False"




print("\n MNIST CON 200 IMAGENES, STRONGLY, 15 LAYERS, 10 LOCAL\n")


n_layers_val = 15

print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
basis = "False"
dense = "False"


print("\n MNIST CON 200 IMAGENES, PAPER MEJORA, 10 LAYERS, 10 LOCAL\n")


n_layers_val = 10


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
basis = "False"
dense = "False"




print("\n MNIST CON 200 IMAGENES, PAPER MEJORA, 15 LAYERS, 10 LOCAL\n")


n_layers_val = 15


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
basis = "False"
dense = "False"


print("\n MNIST CON 200 IMAGENES, PAPER SIN MEJORA, 10 LAYERS, 10 LOCAL\n")


n_layers_val = 10


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
basis = "False"
dense = "False"




print("\n MNIST CON 200 IMAGENES, PAPER SIN MEJORA, 15 LAYERS, 10 LOCAL\n")


n_layers_val = 15


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
basis = "False"
dense = "False"


print("\n SAR CON 80 IMAGENES, STRONGLY, 10 LAYERS, 10 LOCAL\n")


n_layers_val = 10

print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
basis = "False"
dense = "False"



print("\n SAR CON 80 IMAGENES, STRONGLY, 15 LAYERS, 10 LOCAL\n")


n_layers_val = 15

print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
basis = "False"
dense = "False"


print("\n SAR CON 80 IMAGENES, PAPER MEJORA, 10 LAYERS, 10 LOCAL\n")


n_layers_val = 10

print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 80 imagenes, 10 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
basis = "False"
dense = "False"



print("\n SAR CON 80 IMAGENES, PAPER MEJORA, 15 LAYERS, 10 LOCAL\n")


n_layers_val = 15


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 80 imagenes, 15 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
basis = "False"
dense = "False"


print("\n SAR CON 200 IMAGENES, PAPER SIN MEJORA, 10 LAYERS, 10 LOCAL\n")


n_layers_val = 10


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 10 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
basis = "False"
dense = "False"



print("\n SAR CON 200 IMAGENES, PAPER SIN MEJORA, 15 LAYERS, 10 LOCAL\n")


n_layers_val = 15


print("\n Ejecutando autoencoder con Angle, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Basis, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
basis = "True"
mainAngle_BasisAutoencoder.ejecutar_autoencoder(basis, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con Amplitude, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con DenseAngle, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
dense = "True"
mainAmplitude_DenseAutoencoder.ejecutar_autoencoder(dense, dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
print("\n Ejecutando autoencoder con SingleQubit/Reup, ansatz Strongly, 200 imagenes, 15 layers, 10 local ... \n")
mainSQOriginal.ejecutar_autoencoder(dataset, n_global, n_local, n_layers_val, n_train_por_numero, n_test_por_numero, ansatz, mejora)
basis = "False"
dense = "False"