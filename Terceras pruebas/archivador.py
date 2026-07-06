import os
import re
import shutil

def obtener_ruta_disponible(ruta):
    """
    Si la ruta no existe, la devuelve tal cual.
    Si existe, añade _1, _2, _3... antes de la extensión (si es archivo)
    o al final del nombre (si es carpeta).
    """
    if not os.path.exists(ruta):
        return ruta

    directorio = os.path.dirname(ruta)
    nombre = os.path.basename(ruta)

    if os.path.isfile(ruta) or "." in os.path.splitext(nombre)[1]:
        base, ext = os.path.splitext(nombre)

        contador = 1
        while True:
            nueva_ruta = os.path.join(
                directorio,
                f"{base}_{contador}{ext}"
            )

            if not os.path.exists(nueva_ruta):
                return nueva_ruta

            contador += 1
    else:
        contador = 1
        while True:
            nueva_ruta = os.path.join(
                directorio,
                f"{nombre}_{contador}"
            )

            if not os.path.exists(nueva_ruta):
                return nueva_ruta

            contador += 1











# Directorio base
DIRECTORIO_BASE = r"\\datastore.tekniker.es\ia\data-analytics\KUBIBIT\DataEncoding\Resultados\MNIST\StronglyEntangling\Sin Mejora"

# Archivo resumen
# Archivo resumen
ARCHIVO_RESUMEN = obtener_ruta_disponible(
    os.path.join(DIRECTORIO_BASE, "resumen_metricas.txt")
)

# Carpeta para las gráficas
CARPETA_GRAFICAS = obtener_ruta_disponible(
    os.path.join(DIRECTORIO_BASE, "graficas_loss")
)

os.makedirs(CARPETA_GRAFICAS)

# Patrón EXACTO de las carpetas válidas
patron_carpeta = re.compile(
    r"^Amplitude_epochs\d+_batchsize\d+_layers\d+_train\d+_test\d+_bloques\d+_lr\d*\.?\d+$"
)

# Patrones para extraer métricas
patron_mse = re.compile(
    r"MSE medio global de la imagen:\s*([0-9]*\.?[0-9]+)"
)

patron_ssim = re.compile(
    r"SSIM medio de las imagenes:\s*([0-9]*\.?[0-9]+)"
)

resultados = []

for carpeta in os.listdir(DIRECTORIO_BASE):

    ruta_carpeta = os.path.join(DIRECTORIO_BASE, carpeta)

    # Ignorar todo lo que no sea directorio
    if not os.path.isdir(ruta_carpeta):
        continue

    # Ignorar carpetas que no siguen el formato deseado
    if not patron_carpeta.match(carpeta):
        continue

    txt_path = os.path.join(ruta_carpeta, f"{carpeta}.txt")

    if not os.path.isfile(txt_path):
        print(f"TXT no encontrado: {txt_path}")
        continue

    try:

        # Leer el txt
        with open(txt_path, "r", encoding="cp1252") as f:
            contenido = f.read()

        mse_match = patron_mse.search(contenido)
        ssim_match = patron_ssim.search(contenido)

        if mse_match is None or ssim_match is None:
            print(f"No se encontraron métricas en {carpeta}")
            continue

        mse = float(mse_match.group(1))
        ssim = float(ssim_match.group(1))

        resultados.append({
            "carpeta": carpeta,
            "mse": mse,
            "ssim": ssim
        })

        # Buscar train_dev_loss con cualquier extensión
        for archivo in os.listdir(ruta_carpeta):

            nombre, extension = os.path.splitext(archivo)

            if nombre == "train_dev_loss":

                origen = os.path.join(ruta_carpeta, archivo)

                destino = os.path.join(
                    CARPETA_GRAFICAS,
                    f"{carpeta}{extension}"
                )

                shutil.copy2(origen, destino)

                break

    except Exception as e:
        print(f"Error procesando {carpeta}: {e}")

# Ordenar por MSE ascendente
resultados.sort(key=lambda x: x["mse"])

# Guardar resumen
with open(ARCHIVO_RESUMEN, "w", encoding="utf-8") as f:

    f.write("RESULTADOS ORDENADOS POR MSE\n")
    f.write("=" * 120 + "\n\n")

    for r in resultados:

        f.write(f"{r['carpeta']}\n")
        f.write(f"    MSE : {r['mse']:.6f}\n")
        f.write(f"    SSIM: {r['ssim']:.6f}\n\n")

print(f"Resumen guardado en: {ARCHIVO_RESUMEN}")
print(f"Gráficas copiadas en: {CARPETA_GRAFICAS}")
print(f"Experimentos procesados: {len(resultados)}")