**ملاحظة مهمة حول الترجمة**

تمت ترجمة النص أدناه باستخدام أدوات الذكاء الاصطناعي (الترجمة الآلية). نظرًا لإمكانية احتواء هذه العملية على أخطاء أو عدم دقة، نوصي بالرجوع إلى النسخة الأصلية باللغة الإنجليزية أو الإسبانية لضمان دقة المعلومات.

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
يحتوي هذا المستودع على كود لتحويل توقعات YOLO الخاصة بـ Ultralytics إلى تنسيق MOT-Challenge. كما يتضمن سكريبت لإنشاء مجموعة بيانات أرضية للتتبع تركيبية مع نتائج تتبع مطابقة، مما يتيح اختباراً سريعاً باستخدام TrackEval.

## شرح الدوال المساعدة
الوظيفة الأساسية مطبقة في ملف `utils.py`.
يمكنك حفظ التوقعات التي ينتجها نموذج YOLO مباشرةً في تنسيق MOT-Challenge باستخدام دالة `save_mot_results`. وبدلاً من ذلك، إذا كنت تفضل العمل مع ملفات JSON، يمكنك أولاً الحصول على ملف JSON للتوقعات باستخدام دالة `save_results_as_json`، ثم تحويل ذلك الملف إلى تنسيق MOT-Challenge باستخدام `save_mot_from_json`.

### حفظ التوقعات مباشرةً بتنسيق MOT
لحفظ التوقعات، يقوم الكود باسترجاعها باستخدام طريقة `Results.to_df()` لكل إطار، وكتابتها في تنسيق MOT-Challenge:
```txt
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```
في هذا التنسيق، يتم تعيين حقل `id` إلى `-1` للكشفات التي لم تتم متابعتها بعد. بالنسبة لبيانات الحقيقة الأرضية، يمكنك، على سبيل المثال، تعيين قيمة `conf` إلى `1`.
```txt
1,-1,1691.97,381.048,152.23,352.617,0.995616,-1,-1,-1
1,-1,1233.55,467.507,133.65,218.985,0.980069,-1,-1,-1
1,-1,108.484,461.531,97.759,297.453,0.942438,-1,-1,-1
1,-1,256.996,420.694,101.497,296.434,0.938051,-1,-1,-1
```
فيما يلي مثال لإخراج طريقة `to_df()` في كائن نتائج Ultralytics:
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

يمكن لكل من `save_mot_results` و `save_mot_from_json` أن تحفظ أيضاً تسميات التقسيم (segmentation labels) في تنسيق MOTS إذا طلبت ذلك. يُعرّف تنسيق MOTS على النحو التالي:
```txt
<frame> <id> <class_id> <img_height> <img_width> <rle>
```
في هذا التنسيق، يشير `rle` إلى الترميز بطول التشغيل كما هو مستخدم في مجموعة بيانات COCO. وبناءً عليه، فإن مكتبة pycocotools مطلوبة لترميز وفك ترميز الأقنعة في هذا التنسيق.

### حفظ التوقعات بصيغة JSON للاستخدام المستقبلي
يحتوي ملف JSON على التوقعات بصيغة JSON، والتي يتم إنشاؤها باستخدام طريقة `Results.to_json()` لكل إطار.
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

## كيفية استخدام TrackEval
لتقييم نتائج التتبع، ابدأ باستنساخ مستودع [TrackEval](https://github.com/JonathonLuiten/TrackEval/) الأصلي:
```bash
git clone https://github.com/JonathonLuiten/TrackEval.git
```
بعد ذلك، قم بنسخ دليل `data` المُهيأ بشكل صحيح—الذي يجب أن يحتوي على بيانات الحقيقة الأرضية والتوقعات لكل متتبع—إلى دليل TrackEval. لإنشاء مجموعة تركيبية من توقعات المتتبع والحقيقة الأرضية، يمكنك استخدام سكريبت المثال `synthetic_dataset.py`.

بمجرد إعداد كل شيء، قم بتشغيل سكريبت `run_mot_challenge.py` مع المعلمات اللازمة من سطر الأوامر لتقييم مجموعة البيانات المخصصة الخاصة بك. يقوم هذا السكريبت بحساب مقاييس التتبع مثل HOTA و MOTA، وغيرها. لمزيد من التفاصيل، يرجى الرجوع إلى توثيق TrackEval.

**ملاحظة:** تأكد من تعيين `--DO_PREPROC` إلى `False` لتجنب المشكلات المحتملة. على سبيل المثال:
```bash
cd TrackEval
python scripts/run_mot_challenge.py --BENCHMARK MyCustomChallenge --SPLIT_TO_EVAL test --DO_PREPROC False
```

## تنسيق تقييم تحدي MOT
**مسار الحقيقة الأرضية:**  
`/data/gt/mot_challenge/<YourChallenge>-<eval>/<SeqName>/gt/gt.txt`    
يحتوي هذا الملف على تسميات الحقيقة الأرضية بتنسيق MOT-Challenge. هنا، `<YourChallenge>` هو اسم التحدي أو مجموعة البيانات الخاصة بك، و `<eval>` تكون إما `train` أو `test` أو `all`، و `<SeqName>` تُشير إلى تسلسل الفيديو.

يجب أن يكون الهيكلية الدليلية تحت `/data/gt/mot_challenge/<YourChallenge>-<eval>` على النحو التالي:
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

**ملف `seqinfo.ini` مثال:**
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

**ملفات التسلسل:**  
ضمن مجلد `/data/gt/mot_challenge/seqmaps`، قم بإنشاء ملفات نصية تسرد أسماء التسلسلات. على سبيل المثال، أنشئ الملفات `<YourChallenge>-all.txt`، `<YourChallenge>-train.txt`، و `<YourChallenge>-test.txt` بالهيكل التالي:
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

**مسار التوقعات:**  
`/data/trackers/mot_challenge/<YourChallenge>-<eval>/<TrackerName>/data/<SeqName>.txt`   
يخزن هذا الملف التوقعات لكل تسلسل فيديو لكل متتبع. هنا، `<TrackerName>` هو اسم المتتبع أو الخوارزمية المستخدمة.

يجب أن يكون الهيكل الدليلي تحت `/data/trackers/mot_challenge/<YourChallenge>-<eval>` كما يلي:
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
