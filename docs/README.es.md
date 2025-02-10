[![English](https://img.shields.io/badge/lang-English-blue)](README.en.md)
[![Español](https://img.shields.io/badge/lang-Español-purple)](README.es.md)
[![Français](https://img.shields.io/badge/lang-Français-yellow)](README.fr.md)
[![中文](https://img.shields.io/badge/lang-中文-red)](README.zh.md)
[![Português](https://img.shields.io/badge/lang-Português-brightgreen)](README.pt.md)
[![Deutsch](https://img.shields.io/badge/lang-Deutsch-blueviolet)](README.de.md)
[![Italiano](https://img.shields.io/badge/lang-Italiano-orange)](README.it.md)
[![日本語](https://img.shields.io/badge/lang-日本語-yellowgreen)](README.jp.md)
[![العربية](https://img.shields.io/badge/lang-العربية-lightgrey)](README.ar.md)
[![עברית](https://img.shields.io/badge/lang-עברית-teal)](README.he.md)
[![Русский](https://img.shields.io/badge/lang-Русский-lightblue)](README.ru.md)
[![Українська](https://img.shields.io/badge/lang-Українська-skyblue)](README.uk.md)

# YoloMOT

Este repositorio contiene código para convertir las predicciones de Ultralytics YOLO al formato MOT-Challenge. También incluye un script para generar un dataset sintético de *ground truth* para seguimiento junto con resultados de tracking correspondientes, permitiendo pruebas rápidas con TrackEval.

## Explicación de Funciones Utilitarias

La funcionalidad principal está implementada en el archivo `utils.py`.

Puedes guardar predicciones de un modelo YOLO directamente en formato MOT-Challenge usando la función `save_mot_results`. Alternativamente, si prefieres trabajar con archivos JSON, primero puedes obtener un JSON de predicciones usando la función `save_results_as_json`, y luego convertir ese JSON al formato MOT-Challenge con `save_mot_from_json`.

### Guardar Predicciones Directamente en Formato MOT

Para guardar predicciones, el código las obtiene usando el método `Results.to_df()` para cada frame y las escribe en formato MOT-Challenge:
```txt
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```
En este formato, el campo `id` se establece como `-1` para detecciones no rastreadas. Para *ground truth*, puedes establecer `conf` como `1`.

```txt
1,-1,1691.97,381.048,152.23,352.617,0.995616,-1,-1,-1
1,-1,1233.55,467.507,133.65,218.985,0.980069,-1,-1,-1
1,-1,108.484,461.531,97.759,297.453,0.942438,-1,-1,-1
1,-1,256.996,420.694,101.497,296.434,0.938051,-1,-1,-1
```

Ejemplo de salida del método `to_df()` en un objeto Results de Ultralytics:

```bash
       name  class  confidence  \
0    person      0     0.92043   
1    person      0     0.91078   
2    person      0     0.85290   
3  backpack     24     0.60395   

                                                 box  track_id  \
0  {'x1': 1642.54541, 'y1': 597.04321, 'x2': 2352...         1   
1  {'x1': 92.58405, 'y1': 7.53223, 'x2': 727.1163...         2   
2  {'x1': 2227.49854, 'y1': 1024.51343, 'x2': 291...         3   
3  {'x1': 92.28073, 'y1': 203.9021, 'x2': 445.741...         4   

                                            segments  
0  {'x': [2262.0, 2256.0, 2256.0, 2262.0, 2274.0,...  
1  {'x': [228.0, 228.0, 216.0, 210.0, 204.0, 186....  
2  {'x': [2652.0, 2652.0, 2646.0, 2646.0, 2640.0,...  
3  {'x': [204.0, 210.0, 222.0, 222.0, 234.0, 240....  
```

Ambas funciones `save_mot_results` y `save_mot_from_json` también pueden guardar segmentaciones en formato MOTS si se solicita. El formato MOTS se define como:

```txt
<frame> <id> <class_id> <img_height> <img_width> <rle>
```

Aquí, `rle` se refiere a la codificación RLE (run-length encoding) usada een datasets COCO, requiriendo la biblioteca pycocotools.

### Guardar Predicciones como JSON para Uso Futuro

El archivo JSON contiene predicciones generadas con el método `Results.to_json()` por cada frame:

```json
{
    "0": [
        {
            "name": "person",
            "class": 0,
            "confidence": 0.91342,
            "box": {
                "x1": 2751.35596,
                "y1": 243.29077,
                "x2": 3436.21362,
                "y2": 2054.81641
            },
            "track_id": 1,
            "segments": {
                "x": [2868.0, 2868.0, 2862.0, ..., 3048.0],
                "y": [252.0, 270.0, 276.0, ..., 252.0]
            }
        },
        {
            "name": "skateboard",
            "class": 36,
            "confidence": 0.39874,
            "box": {
                "x1": 3746.27856,
                "y1": 1691.79749,
                "x2": 3838.45679,
                "y2": 1793.54187
            },
            "track_id": 5,
            "segments": {
                "x": [3756.0, 3756.0, 3774.0, ..., 3834.0],
                "y": [1692.0, 1734.0, 1734.0, ..., 1692.0]
            }
        }
    ],
    "1": [
        {
            "name": "person",
            "class": 0,
            "confidence": 0.93321,
            "box": {
                "x1": 3064.85596,
                "y1": 260.86932,
                "x2": 3836.20898,
                "y2": 2044.81006
            },
            ...
        }
    ],
    ...
}
```

## Cómo Usar TrackEval

Para evaluar resultados de tracking, empieza por clonar el repositorio original de [TrackEval](https://github.com/JonathonLuiten/TrackEval/):

```bash
git clone https://github.com/JonathonLuiten/TrackEval.git
```

A continuación, copia el directorio `data` correctamente configurado -que debiese contener los datos de _ground truth_ y las predicciones de cada tracker. Para crear un dataset sintético de y _ground truths_, puedes utilizar el script de ejemplo `synthetic_dataset.py`.

Una vez que tengas todo configurado, ejecuta el script `run_mot_challenge.py` con los argumentos de línea de comandos correspondientes para evaluar tu dataset personalizado. Este script calcula métricas de seguimiento como HOTA, MOTA, etc. Para obtener más información, consulte la documentación de TrackEval.

**Nota:** Asegúrate de establecer `--DO_PREPROC` en `False` para evitar posibles problemas. Por ejemplo:

```bash
cd TrackEval
python scripts/run_mot_challenge.py --BENCHMARK MyCustomChallenge --SPLIT_TO_EVAL test --DO_PREPROC False
```

## Formato de Evaluación MOT Challenge

**Estructura de Directorios para *Ground Truth*:**      
`/data/gt/mot_challenge/<YourChallenge>-<eval>/<SeqName>/gt/gt.txt`     
Este archivo contiene el *ground truth* en formato MOT. Aquí, `<YourChallenge>` es el nombre del dataset, `<eval>` puede ser `train`, `test`, o `all` y `<SeqName>` corresponde al video o secuencia.

La estructura de las carpetas y archivos dentro de `/data/gt/mot_challenge/<YourChallenge>-<eval>` debiese ser la siguiente:

```
.
|—— <SeqName01>
    |—— gt
        |—— gt.txt
    |—— seqinfo.ini
|—— <SeqName02>
    |—— ...
|—— <SeqName03>
    |—— ...
```

**Archivo `seqinfo.ini` de ejemplo:**

```
[Sequence]
name=<SeqName>
imDir=img1
frameRate=30
seqLength=525
imWidth=1920
imHeight=1080
imExt=.jpg
```

**Archivos de Secuencias:**
Dentro del directorio `/data/gt/mot_challenge/seqmaps`, deberás crear archivos de texto donde enlistes las secuencias de video correspondiente a esa subsección de tu dataset. Por ejemplo los archivos `<YourChallenge>-all.txt`, `<YourChallenge>-train.txt` y `<YourChallenge>-test.txt`con el siguiente formato:

```
<YourChallenge>-all.txt
name
<seqName1> 
<seqName2>
<seqName3>

<YourChallenge>-train.txt
name
<seqName1> 
<seqName2>

<YourChallenge>-test.txt
name
<seqName3>
```

**Predicciones:**  
`/data/trackers/mot_challenge/<YourChallenge>-<eval>/<TrackerName>/data/<SeqName>.txt`      
Este archivo guarda las predicciones de cada secuencia de video por cada tracker. Donde `<TrackerName>` es el nombre del tracker usado.

La estructura de las carpetas y archivos dentro de `/data/trackers/mot_challenge/<YourChallenge>-<eval>` debiese ser la siguiente:

```
.
|—— <Tracker01>
    |—— data
        |—— <SeqName01>.txt
        |—— <SeqName02>.txt
|—— <Tracker02>
    |—— ...
|—— <Tracker03>
    |—— ...
```
