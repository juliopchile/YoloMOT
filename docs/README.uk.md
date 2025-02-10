**Важлива примітка щодо перекладу**

Нижче наведено текст, перекладений за допомогою інструментів штучного інтелекту (автоматичний переклад). Оскільки цей процес може містити помилки або неточності, радимо звернутися до оригіналу англійською чи іспанською мовами для забезпечення точності інформації.

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
Цей репозиторій містить код для перетворення прогнозів Ultralytics YOLO у формат MOT-Challenge. Також він включає скрипт для генерації синтетичного набору даних з істинними значеннями відстеження разом з відповідними результатами відстеження, що дозволяє швидко тестувати з TrackEval.

## Пояснення корисних функцій
Основна функціональність реалізована у файлі `utils.py`.

Ви можете зберегти прогнози, отримані від моделі YOLO, безпосередньо у форматі MOT-Challenge, використовуючи функцію `save_mot_results`. Або, якщо ви бажаєте працювати з JSON файлами, спочатку можна отримати JSON файл з прогнозами за допомогою функції `save_results_as_json`, а потім перетворити цей JSON у формат MOT-Challenge за допомогою `save_mot_from_json`.

### Збереження прогнозів безпосередньо у формат MOT
Щоб зберегти прогнози, код отримує їх, використовуючи метод `Results.to_df()` для кожного кадру, та записує їх у форматі MOT-Challenge:
```txt
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```
У цьому форматі поле `id` встановлено як `-1` для виявлених об'єктів, які ще не були відслідковані. Для даних істинних значень (ground truth) можна, наприклад, встановити значення `conf` рівним `1`.
```txt
1,-1,1691.97,381.048,152.23,352.617,0.995616,-1,-1,-1
1,-1,1233.55,467.507,133.65,218.985,0.980069,-1,-1,-1
1,-1,108.484,461.531,97.759,297.453,0.942438,-1,-1,-1
1,-1,256.996,420.694,101.497,296.434,0.938051,-1,-1,-1
```
Нижче наведено приклад виводу при використанні методу `to_df()` в об'єкті Ultralytics Results:
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

Функції `save_mot_results` та `save_mot_from_json` також можуть зберігати мітки сегментації у форматі MOTS, якщо це необхідно. Формат MOTS визначено наступним чином:
```txt
<frame> <id> <class_id> <img_height> <img_width> <rle>
```
У цьому форматі `rle` означає кодування за методом run-length, яке використовується в наборі даних COCO. Відповідно, бібліотека pycocotools необхідна для кодування та декодування масок у цьому форматі.

### Збереження прогнозів у форматі JSON для подальшого використання
JSON файл містить прогнози у форматі JSON, згенеровані за допомогою методу `Results.to_json()` для кожного кадру.
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

## Як використовувати TrackEval
Щоб оцінити результати відстеження, розпочніть з клонування оригінального репозиторію [TrackEval](https://github.com/JonathonLuiten/TrackEval/):
```bash
git clone https://github.com/JonathonLuiten/TrackEval.git
```
Далі, скопіюйте правильно налаштований каталог `data` —який повинен містити ваші істинні значення та прогнози для кожного трекера— у каталог TrackEval. Щоб створити синтетичну колекцію прогнозів трекера та істинних значень, ви можете скористатися прикладом скрипта `synthetic_dataset.py`.
Як тільки все буде налаштовано, запустіть скрипт `run_mot_challenge.py` з відповідними аргументами командного рядка для оцінки вашого користувацького набору даних. Цей скрипт обчислює метрики відстеження, такі як HOTA, MOTA тощо. Для отримання додаткової інформації зверніться до документації TrackEval.
**Примітка:** Переконайтеся, що ви встановили `--DO_PREPROC` у значення `False`, щоб уникнути можливих проблем. Наприклад:
```bash
cd TrackEval
python scripts/run_mot_challenge.py --BENCHMARK MyCustomChallenge --SPLIT_TO_EVAL test --DO_PREPROC False
```

## Формат оцінки MOT Challenge
**Шлях до істинних значень:**   
`/data/gt/mot_challenge/<YourChallenge>-<eval>/<SeqName>/gt/gt.txt`     
Цей файл містить мітки істинних значень у форматі MOT-Challenge. Тут `<YourChallenge>` є назвою вашого випробування або набору даних, `<eval>` може бути `train`, `test` або `all`, а `<SeqName>` відповідає імені відеопослідовності.

Структура директорій у каталозі `/data/gt/mot_challenge/<YourChallenge>-<eval>` повинна бути наступною:
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

**Приклад файлу `seqinfo.ini`:**
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

**Файли послідовностей:**   
У папці `/data/gt/mot_challenge/seqmaps` створіть текстові файли з переліком імен послідовностей. Наприклад, створіть файли `<YourChallenge>-all.txt`, `<YourChallenge>-train.txt` та `<YourChallenge>-test.txt` з наступною структурою:
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

**Шлях до прогнозів:**   
`/data/trackers/mot_challenge/<YourChallenge>-<eval>/<TrackerName>/data/<SeqName>.txt`   
Цей файл зберігає прогнози для кожної відеопослідовності для кожного трекера. Тут `<TrackerName>` є назвою трекера або алгоритму, який використовується.

Структура директорій у каталозі `/data/trackers/mot_challenge/<YourChallenge>-<eval>` повинна бути наступною:
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