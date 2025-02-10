**Wichtiger Hinweis zur Übersetzung**

Der nachfolgende Text wurde mithilfe von KI-Übersetzungstools (automatische Übersetzung) übersetzt. Da dieser Vorgang Fehler oder Ungenauigkeiten enthalten kann, empfehlen wir, zur Gewährleistung der Genauigkeit die Originalversion in Englisch oder Spanisch heranzuziehen.

---

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
Dieses Repository enthält Code, um Ultralytics YOLO-Vorhersagen in das MOT-Challenge-Format zu konvertieren. Außerdem beinhaltet es ein Skript, um ein synthetisches Tracking-„Ground Truth“-Dataset zusammen mit entsprechenden Tracking-Ergebnissen zu generieren, was schnelle Tests mit TrackEval ermöglicht.

## Erklärung der Hilfsfunktionen
Die Kernfunktionalität ist in der Datei `utils.py` implementiert.

Sie können die von einem YOLO-Modell erzeugten Vorhersagen direkt im MOT-Challenge-Format mit der Funktion `save_mot_results` speichern. Alternativ, wenn Sie lieber mit JSON-Dateien arbeiten, können Sie zunächst eine JSON-Datei der Vorhersagen mit der Funktion `save_results_as_json` erstellen und diese JSON anschließend mit `save_mot_from_json` in das MOT-Challenge-Format konvertieren.

### Vorhersagen direkt im MOT-Format speichern
Um Vorhersagen zu speichern, ruft der Code diese für jeden Frame mit der Methode `Results.to_df()` ab und schreibt sie im MOT-Challenge-Format:

```txt
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```

In diesem Format wird das Feld `id` auf `-1` gesetzt für Detektionen, die noch nicht verfolgt wurden. Bei Ground-Truth-Daten können Sie beispielsweise den `conf`-Wert auf `1` setzen.

```txt
1,-1,1691.97,381.048,152.23,352.617,0.995616,-1,-1,-1
1,-1,1233.55,467.507,133.65,218.985,0.980069,-1,-1,-1
1,-1,108.484,461.531,97.759,297.453,0.942438,-1,-1,-1
1,-1,256.996,420.694,101.497,296.434,0.938051,-1,-1,-1
```

Nachfolgend ein Beispielauszug, der die Ausgabe der Methode `to_df()` in einem Ultralytics Results-Objekt zeigt:

```bash
       name    class  confidence  \
0     person      0     0.92043   
1     person      0     0.91078   
2     person      0     0.85290   
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

Sowohl `save_mot_results` als auch `save_mot_from_json` können, falls gewünscht, auch Segmentierungslabels im MOTS-Format speichern. Das MOTS-Format ist wie folgt definiert:

```txt
<frame> <id> <class_id> <img_height> <img_width> <rle>
```

In diesem Format steht `rle` für Run-Length Encoding, wie es im COCO-Dataset verwendet wird. Folglich wird die pycocotools-Bibliothek benötigt, um Masken in diesem Format zu kodieren und zu dekodieren.

### Vorhersagen als JSON für die zukünftige Nutzung speichern
Die JSON-Datei enthält die Vorhersagen im JSON-Format, welche für jeden Frame mit der Methode `Results.to_json()` generiert werden.

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

## Wie man TrackEval benutzt

Um Tracking-Ergebnisse zu evaluieren, beginnen Sie damit, das originale [TrackEval](https://github.com/JonathonLuiten/TrackEval/) Repository zu klonen:

```bash
git clone https://github.com/JonathonLuiten/TrackEval.git
```

Anschließend kopieren Sie das korrekt konfigurierte `data`-Verzeichnis – welches Ihre Ground-Truth-Daten sowie Vorhersagen für jeden Tracker enthalten sollte – in das TrackEval-Verzeichnis. Um eine synthetische Sammlung von Tracker-Vorhersagen und Ground-Truths zu erstellen, können Sie das Beispiels-Skript `synthetic_dataset.py` verwenden.

Sobald alles eingerichtet ist, führen Sie das Skript `run_mot_challenge.py` mit den entsprechenden Befehlszeilenargumenten aus, um Ihr benutzerdefiniertes Dataset zu evaluieren. Dieses Skript berechnet Tracking-Metriken wie HOTA, MOTA etc. Weitere Details finden Sie in der TrackEval-Dokumentation.

**Hinweis:** Stellen Sie sicher, dass Sie `--DO_PREPROC` auf `False` setzen, um mögliche Probleme zu vermeiden. Zum Beispiel:

```bash
cd TrackEval
python scripts/run_mot_challenge.py --BENCHMARK MyCustomChallenge --SPLIT_TO_EVAL test --DO_PREPROC False
```

## MOT Challenge Evaluierungsformat

**Pfad zur Ground Truth:**  
`/data/gt/mot_challenge/<YourChallenge>-<eval>/<SeqName>/gt/gt.txt`  
Diese Datei enthält die Ground-Truth-Labels im MOT-Challenge-Format. Hierbei ist `<YourChallenge>` der Name Ihrer Challenge oder Ihres Datasets, `<eval>` entweder `train`, `test` oder `all`, und `<SeqName>` entspricht der Video-Sequenz.

Die Verzeichnisstruktur unter `/data/gt/mot_challenge/<YourChallenge>-<eval>` sollte wie folgt aussehen:

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

**Beispiel einer `seqinfo.ini`-Datei:**

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

**Sequenzdateien:**  
Erstellen Sie im Ordner `/data/gt/mot_challenge/seqmaps` Textdateien, die die Sequenznamen auflisten. Zum Beispiel erstellen Sie `<YourChallenge>-all.txt`, `<YourChallenge>-train.txt` und `<YourChallenge>-test.txt` mit folgender Struktur:

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

**Pfad der Vorhersagen:**  
`/data/trackers/mot_challenge/<YourChallenge>-<eval>/<TrackerName>/data/<SeqName>.txt`  
Diese Datei speichert die Vorhersagen für jede Video-Sequenz pro Tracker. Hierbei ist `<TrackerName>` der Name des verwendeten Trackers oder Algorithmus.

Die Verzeichnisstruktur unter `/data/trackers/mot_challenge/<YourChallenge>-<eval>` sollte wie folgt aussehen:

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
