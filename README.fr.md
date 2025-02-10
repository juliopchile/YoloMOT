**Note importante sur la traduction**.

Le texte ci-dessous a été traduit à l'aide d'outils d'IA (traduction automatique). Comme ce processus peut comporter des erreurs ou des imprécisions, nous recommandons de consulter la version originale en anglais ou en espagnol afin de garantir l'exactitude des informations.

---
[![English](https://img.shields.io/badge/lang-English-blue)](docs/README.en.md)
[![Español](https://img.shields.io/badge/lang-Español-purple)](docs/README.es.md)
[![Français](https://img.shields.io/badge/lang-Français-yellow)](docs/README.fr.md)
[![中文](https://img.shields.io/badge/lang-中文-red)](docs/README.zh.md)
[![Português](https://img.shields.io/badge/lang-Português-brightgreen)](docs/README.pt.md)
[![Deutsch](https://img.shields.io/badge/lang-Deutsch-blueviolet)](docs/README.de.md)
[![Italiano](https://img.shields.io/badge/lang-Italiano-orange)](docs/README.it.md)
[![日本語](https://img.shields.io/badge/lang-日本語-yellowgreen)](docs/README.jp.md)
[![العربية](https://img.shields.io/badge/lang-العربية-lightgrey)](docs/README.ar.md)
[![עברית](https://img.shields.io/badge/lang-עברית-teal)](docs/README.he.md)



# YoloMOT

Ce dépôt contient du code permettant de convertir les prédictions d'Ultralytics YOLO au format MOT-Challenge. Il inclut également un script pour générer un jeu de données synthétique de vérité terrain pour le suivi, ainsi que les résultats de tracking correspondants, facilitant ainsi les tests rapides avec TrackEval.

## Explication des fonctions utilitaires

La fonctionnalité principale est implémentée dans le fichier `utils.py`.

Vous pouvez sauvegarder directement les prédictions produites par un modèle YOLO au format MOT-Challenge en utilisant la fonction `save_mot_results`. Sinon, si vous préférez travailler avec des fichiers JSON, vous pouvez d'abord obtenir un fichier JSON de prédictions via la fonction `save_results_as_json`, puis convertir ce JSON au format MOT-Challenge avec `save_mot_from_json`.

### Sauvegarder les prédictions directement au format MOT

Pour sauvegarder les prédictions, le code les récupère grâce à la méthode `Results.to_df()` pour chaque frame, puis les écrit au format MOT-Challenge :

```txt
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```

Dans ce format, le champ `id` est fixé à `-1` pour les détections qui n'ont pas encore été trackées. Pour les données de vérité terrain, vous pouvez, par exemple, définir la valeur de `conf` à `1`.

```txt
1,-1,1691.97,381.048,152.23,352.617,0.995616,-1,-1,-1
1,-1,1233.55,467.507,133.65,218.985,0.980069,-1,-1,-1
1,-1,108.484,461.531,97.759,297.453,0.942438,-1,-1,-1
1,-1,256.996,420.694,101.497,296.434,0.938051,-1,-1,-1
```

Voici un exemple de sortie obtenu en utilisant la méthode `to_df()` sur un objet Results d'Ultralytics :

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

Les fonctions `save_mot_results` et `save_mot_from_json` peuvent également sauvegarder les labels de segmentation au format MOTS si besoin. Le format MOTS est défini comme suit :

```txt
<frame> <id> <class_id> <img_height> <img_width> <rle>
```

Dans ce format, `rle` signifie "run-length encoding" tel qu'utilisé dans le dataset COCO. Par conséquent, la bibliothèque pycocotools est nécessaire pour encoder et décoder les masques dans ce format.

### Sauvegarder les prédictions au format JSON pour une utilisation ultérieure

Le fichier JSON contient les prédictions au format JSON, générées à l'aide de la méthode `Results.to_json()` pour chaque frame.

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

## Comment utiliser TrackEval

Pour évaluer vos résultats de tracking, commencez par cloner le dépôt original de [TrackEval](https://github.com/JonathonLuiten/TrackEval/):

```bash
git clone https://github.com/JonathonLuiten/TrackEval.git
```

Ensuite, copiez le répertoire `data` correctement configuré — qui doit contenir vos données de vérité terrain et les prédictions pour chaque tracker — dans le dossier de TrackEval. Pour créer une collection synthétique de prédictions de trackers et de vérités terrain, vous pouvez utiliser le script d'exemple `synthetic_dataset.py`.

Une fois tout en place, exécutez le script `run_mot_challenge.py` avec les arguments en ligne de commande appropriés pour évaluer votre dataset personnalisé. Ce script calcule des métriques de suivi telles que HOTA, MOTA, etc. Pour plus de détails, consultez la documentation de TrackEval.

**Remarque :** Assurez-vous de définir `--DO_PREPROC` à `False` pour éviter d'éventuels problèmes. Par exemple :

```bash
cd TrackEval
python scripts/run_mot_challenge.py --BENCHMARK MyCustomChallenge --SPLIT_TO_EVAL test --DO_PREPROC False
```

## Format d'évaluation MOT Challenge

**Chemin de la vérité terrain :**  
`/data/gt/mot_challenge/<YourChallenge>-<eval>/<SeqName>/gt/gt.txt`  
Ce fichier contient les labels de vérité terrain au format MOT-Challenge. Ici, `<YourChallenge>` est le nom de votre challenge ou dataset, `<eval>` peut être `train`, `test` ou `all`, et `<SeqName>` correspond à la séquence vidéo.

La structure du répertoire sous `/data/gt/mot_challenge/<YourChallenge>-<eval>` doit être la suivante :

```
.
|—— <SeqName01>
    |—— gt
        |—— gt.txt
    |—— seqinfo.ini
|—— <SeqName02>
    |—— ……
|—— <SeqName03>
    |—— …...
```

**Exemple de fichier `seqinfo.ini` :**

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

**Fichiers de séquences :**  
Dans le dossier `/data/gt/mot_challenge/seqmaps`, créez des fichiers texte listant les noms des séquences. Par exemple, créez `<YourChallenge>-all.txt`, `<YourChallenge>-train.txt` et `<YourChallenge>-test.txt` avec la structure suivante :

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

**Chemin des prédictions :**  
`/data/trackers/mot_challenge/<YourChallenge>-<eval>/<TrackerName>/data/<SeqName>.txt`  
Ce fichier contient les prédictions pour chaque séquence vidéo, pour chaque tracker. Ici, `<TrackerName>` correspond au nom du tracker ou de l'algorithme utilisé.

La structure du répertoire sous `/data/trackers/mot_challenge/<YourChallenge>-<eval>` doit être la suivante :

```
.
|—— <Tracker01>
    |—— data
        |—— <SeqName01>.txt
        |—— <SeqName02>.txt
|—— <Tracker02>
    |—— ……
|—— <Tracker03>
    |—— …...
```