**Важное примечание относительно перевода**

Ниже приведён текст, переведённый с использованием инструментов искусственного интеллекта (автоматический перевод). Поскольку этот процесс может содержать ошибки или неточности, рекомендуется обратиться к оригиналу на английском или испанском языках для обеспечения точности информации.

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
Этот репозиторий содержит код для преобразования предсказаний Ultralytics YOLO в формат MOT-Challenge. Также он включает скрипт для генерации синтетического набора данных с эталонными (ground truth) метками для трекинга вместе с соответствующими результатами трекинга, что позволяет быстро протестировать с помощью TrackEval.

## Объяснение утилитарных функций
Основная функциональность реализована в файле `utils.py`.
Вы можете сохранить предсказания, полученные от модели YOLO, непосредственно в формате MOT-Challenge с помощью функции `save_mot_results`. Либо, если вам удобнее работать с JSON-файлами, вы можете сначала получить JSON с предсказаниями, используя функцию `save_results_as_json`, а затем преобразовать этот JSON в формат MOT-Challenge с помощью функции `save_mot_from_json`.

### Сохранение предсказаний непосредственно в формате MOT
Для сохранения предсказаний код извлекает их с помощью метода `Results.to_df()` для каждого кадра и записывает их в формате MOT-Challenge:
```txt
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```
В этом формате поле `id` устанавливается в `-1` для обнаружений, которые еще не были отслежены. Для эталонных данных, например, можно задать значение `conf` равным `1`.
```txt
1,-1,1691.97,381.048,152.23,352.617,0.995616,-1,-1,-1
1,-1,1233.55,467.507,133.65,218.985,0.980069,-1,-1,-1
1,-1,108.484,461.531,97.759,297.453,0.942438,-1,-1,-1
1,-1,256.996,420.694,101.497,296.434,0.938051,-1,-1,-1
```
Ниже приведён пример вывода, полученного при использовании метода `to_df()` в объекте Ultralytics Results:
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

Обе функции `save_mot_results` и `save_mot_from_json` могут также сохранить метки сегментации в формате MOTS по запросу. Формат MOTS определяется следующим образом:
```txt
<frame> <id> <class_id> <img_height> <img_width> <rle>
```
В этом формате `rle` означает кодирование с помощью пробега длины (run-length encoding), как используется в наборе данных COCO. Соответственно, библиотека pycocotools требуется для кодирования и декодирования масок в этом формате.

### Сохранение предсказаний в виде JSON для последующего использования
JSON-файл содержит предсказания в формате JSON, сгенерированные с помощью метода `Results.to_json()` для каждого кадра.
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

## Как использовать TrackEval
Для оценки результатов трекинга начните с клонирования оригинального репозитория [TrackEval](https://github.com/JonathonLuiten/TrackEval/):
```bash
git clone https://github.com/JonathonLuiten/TrackEval.git
```
Затем скопируйте корректно настроенную директорию `data` — она должна содержать ваши эталонные данные и предсказания для каждого трекера — в директорию TrackEval. Чтобы создать синтетический набор предсказаний трекера и эталонных меток, вы можете использовать пример скрипта `synthetic_dataset.py`.
После того как всё настроено, запустите скрипт `run_mot_challenge.py` с соответствующими аргументами командной строки для оценки вашего кастомного набора данных. Этот скрипт вычисляет метрики трекинга, такие как HOTA, MOTA и т.д. Для получения дополнительной информации обратитесь к документации TrackEval.

**Примечание:** Убедитесь, что вы установили `--DO_PREPROC` в значение `False`, чтобы избежать возможных проблем. Например:
```bash
cd TrackEval
python scripts/run_mot_challenge.py --BENCHMARK MyCustomChallenge --SPLIT_TO_EVAL test --DO_PREPROC False
```

## Формат оценки MOT Challenge
**Путь к эталонным данным (Ground Truth):**  
`/data/gt/mot_challenge/<YourChallenge>-<eval>/<SeqName>/gt/gt.txt`  
Этот файл содержит эталонные метки в формате MOT-Challenge. Здесь `<YourChallenge>` — это название вашего челленджа или набора данных, `<eval>` может быть `train`, `test` или `all`, а `<SeqName>` соответствует названию видеопоследовательности.

Структура директорий внутри `/data/gt/mot_challenge/<YourChallenge>-<eval>` должна выглядеть следующим образом:
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

**Пример файла `seqinfo.ini`:**
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

**Файлы последовательностей (Sequence Files):**  
В папке `/data/gt/mot_challenge/seqmaps` создайте текстовые файлы со списком названий последовательностей. Например, создайте файлы `<YourChallenge>-all.txt`, `<YourChallenge>-train.txt` и `<YourChallenge>-test.txt` со следующей структурой:
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

**Путь к предсказаниям:**  
`/data/trackers/mot_challenge/<YourChallenge>-<eval>/<TrackerName>/data/<SeqName>.txt`  
Этот файл содержит предсказания для каждой видеопоследовательности для каждого трекера. Здесь `<TrackerName>` — это название трекера или алгоритма, используемого для трекинга.

Структура директорий внутри `/data/trackers/mot_challenge/<YourChallenge>-<eval>` должна выглядеть следующим образом:
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