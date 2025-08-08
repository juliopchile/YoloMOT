import os

# Rutas de las carpetas
carpeta_datos = "dataset/cvat_labels"
carpeta_media = "dataset/source"

# Obtener nombres de archivos
archivos_datos = {os.path.splitext(f)[0]: f for f in os.listdir(carpeta_datos)}
archivos_media = {os.path.splitext(f)[0]: f for f in os.listdir(carpeta_media)}

# Emparejar por nombre base
diccionario = {}
for nombre in archivos_datos.keys() & archivos_media.keys():
    diccionario[nombre] = (archivos_datos[nombre], archivos_media[nombre])

print(diccionario)
