import os
import re
import pandas as pd
import matplotlib.pyplot as plt

DIRECTORIO_BASE = r"\\datastore.tekniker.es\ia\data-analytics\KUBIBIT\DataEncoding\Resultados\MNIST\StronglyEntangling\Sin Mejora"

patron_nombre = re.compile(
    r"Amplitude_epochs(?P<epochs>\d+)"
    r"_batchsize(?P<batch>\d+)"
    r"_layers(?P<layers>\d+)"
    r"_train(?P<train>\d+)"
    r"_test(?P<test>\d+)"
    r"_bloques(?P<k>\d+)"
    r"_lr(?P<lr>\d*\.?\d+)"
)

patron_mse = re.compile(
    r"MSE medio global de la imagen:\s*([0-9]*\.?[0-9]+)"
)

datos = []

for carpeta in os.listdir(DIRECTORIO_BASE):

    ruta_carpeta = os.path.join(DIRECTORIO_BASE, carpeta)

    if not os.path.isdir(ruta_carpeta):
        continue

    match = patron_nombre.match(carpeta)

    if not match:
        continue

    txt_path = os.path.join(ruta_carpeta, f"{carpeta}.txt")

    if not os.path.exists(txt_path):
        continue

    with open(txt_path, "r", encoding="cp1252") as f:
        contenido = f.read()

    mse_match = patron_mse.search(contenido)

    if mse_match is None:
        continue

    datos.append({
        "batch": int(match.group("batch")),
        "layers": int(match.group("layers")),
        "k": int(match.group("k")),
        "lr": float(match.group("lr")),
        "mse": float(mse_match.group(1))
    })

df = pd.DataFrame(datos)

print(df.head())


print("\nMSE medio por LR")
print(df.groupby("lr")["mse"].mean().sort_values())

print("\nMSE medio por Layers")
print(df.groupby("layers")["mse"].mean().sort_values())

print("\nMSE medio por Batch")
print(df.groupby("batch")["mse"].mean().sort_values())

print("\nMSE medio por K")
print(df.groupby("k")["mse"].mean().sort_values())

print(
    df.groupby(["lr", "layers"])["mse"]
    .mean()
    .sort_values()
)


print(
    df.sort_values("mse")
      .head(10)
)

print(df.sort_values("mse").head(10)["lr"].value_counts())



df.boxplot(column="mse", by="lr")

plt.title("Distribución del MSE según Learning Rate")
plt.suptitle("")
plt.ylabel("MSE")
plt.show()