**Nota importante sulla traduzione**

Il testo seguente è stato tradotto mediante strumenti di intelligenza artificiale (traduzione automatica). Poiché questo processo potrebbe contenere errori o imprecisioni, si consiglia di consultare la versione originale in inglese o in spagnolo per garantire l’accuratezza delle informazioni.

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
Questo repository contiene il codice per convertire le previsioni di Ultralytics YOLO nel formato MOT-Challenge. Include inoltre uno script per generare un dataset sintetico di dati di verità di tracciamento insieme ai relativi risultati di tracking, permettendo test rapidi con TrackEval.

## Spiegazione delle Funzioni di Utilità
La funzionalità principale è implementata nel file `utils.py`.

Puoi salvare le previsioni prodotte da un modello YOLO direttamente nel formato MOT-Challenge usando la funzione `save_mot_results`. In alternativa, se preferisci lavorare con file JSON, puoi prima ottenere un file JSON di previsioni usando la funzione `save_results_as_json`, per poi convertire quel JSON nel formato MOT-Challenge con `save_mot_from_json`.

### Salva le Previsioni Direttamente nel Formato MOT
Per salvare le previsioni, il codice le recupera utilizzando il metodo `Results.to_df()` per ogni frame e le scrive nel formato MOT-Challenge:
```txt
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```
In questo formato, il campo `id` viene impostato a `-1` per le detezioni che non sono ancora state tracciate. Per i dati di verità, puoi, ad esempio, impostare il valore `conf` a `1`.
```txt
1,-1,1691.97,381.048,152.23,352.617,0.995616,-1,-1,-1
1,-1,1233.55,467.507,133.65,218.985,0.980069,-1,-1,-1
1,-1,108.484,461.531,97.759,297.453,0.942438,-1,-1,-1
1,-1,256.996,420.694,101.497,296.434,0.938051,-1,-1,-1
```
Di seguito è riportato un esempio di output dall'uso del metodo `to_df()` in un oggetto Results di Ultralytics:
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
Sia `save_mot_results` che `save_mot_from_json` possono anche salvare le etichette di segmentazione nel formato MOTS, se richiesto. Il formato MOTS è definito come segue:
```txt
<frame> <id> <class_id> <img_height> <img_width> <rle>
```
In questo formato, `rle` sta per run-length encoding, come usato nel dataset COCO. Di conseguenza, la libreria pycocotools è necessaria per codificare e decodificare le maschere in questo formato.

### Salva le Previsioni in JSON per un Futuro Utilizzo
Il file JSON contiene le previsioni in formato JSON, generate usando il metodo `Results.to_json()` per ogni frame.
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

## Come Utilizzare TrackEval
Per valutare i risultati del tracking, inizia clonando il repository originale di [TrackEval](https://github.com/JonathonLuiten/TrackEval/):
```bash
git clone https://github.com/JonathonLuiten/TrackEval.git
```

Successivamente, copia la directory `data` opportunamente configurata — che dovrebbe contenere i tuoi dati di verità e le previsioni per ciascun tracker — nella directory di TrackEval. Per creare una raccolta sintetica di previsioni tracker e ground truths, puoi usare lo script di esempio `synthetic_dataset.py`.

Una volta configurato tutto, esegui lo script `run_mot_challenge.py` con i relativi argomenti da linea di comando per valutare il tuo dataset personalizzato. Questo script calcola metriche di tracking come HOTA, MOTA, ecc. Per ulteriori dettagli, consulta la documentazione di TrackEval.

**Nota:** Assicurati di impostare `--DO_PREPROC` su `False` per evitare potenziali problemi. Ad esempio:
```bash
cd TrackEval
python scripts/run_mot_challenge.py --BENCHMARK MyCustomChallenge --SPLIT_TO_EVAL test --DO_PREPROC False
```

## Formato di Valutazione MOT Challenge
**Percorso Ground Truth:**  
`/data/gt/mot_challenge/<YourChallenge>-<eval>/<SeqName>/gt/gt.txt`    
Questo file contiene le etichette di ground truth nel formato MOT-Challenge. Qui, `<YourChallenge>` è il nome della tua sfida o dataset, `<eval>` può essere `train`, `test` o `all`, e `<SeqName>` corrisponde alla sequenza video.

La struttura delle directory sotto `/data/gt/mot_challenge/<YourChallenge>-<eval>` dovrebbe essere la seguente:
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

**Esempio di file `seqinfo.ini`:**
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

**File di Sequenza:**  
All'interno della cartella `/data/gt/mot_challenge/seqmaps`, crea dei file di testo che elencano i nomi delle sequenze. Ad esempio, crea `<YourChallenge>-all.txt`, `<YourChallenge>-train.txt` e `<YourChallenge>-test.txt` con la seguente struttura:
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

**Percorso Previsioni:**  
`/data/trackers/mot_challenge/<YourChallenge>-<eval>/<TrackerName>/data/<SeqName>.txt`   
Questo file memorizza le previsioni per ogni sequenza video per ciascun tracker. Qui, `<TrackerName>` è il nome del tracker o dell'algoritmo utilizzato.

La struttura delle directory sotto `/data/trackers/mot_challenge/<YourChallenge>-<eval>` dovrebbe essere la seguente:
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
