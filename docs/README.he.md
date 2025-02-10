**הערה חשובה לגבי התרגום**

הטקסט הבא תורגם באמצעות כלים של בינה מלאכותית (תרגום אוטומטי). מכיוון שתהליך זה עלול לכלול שגיאות או חוסר דיוקים, אנו ממליצים לעיין בגרסה המקורית באנגלית או בספרדית כדי להבטיח את דיוק המידע.

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
רפרוזיטורי זה מכיל קוד להמרת תחזיות Ultralytics YOLO לפורמט MOT-Challenge. הוא כולל גם סקריפט ליצירת מערכת נתונים סינתטית של אמת טרקינג יחד עם תוצאות טרקינג תואמות, מה שמאפשר בדיקה מהירה עם TrackEval.

## הסבר על פונקציות עזר
התפקוד הליבה מיושם בקובץ `utils.py`.
ניתן לשמור תחזיות שמופקות על ידי מודל YOLO ישירות בפורמט MOT-Challenge באמצעות הפונקציה `save_mot_results`. לחלופין, אם אתה מעדיף לעבוד עם קבצי JSON, תוכל תחילה לקבל קובץ JSON של תחזיות באמצעות הפונקציה `save_results_as_json`, ואז להפוך את ה-JSON לפורמט MOT-Challenge עם הפונקציה `save_mot_from_json`.

### שמירת תחזיות ישירות לפורמט MOT
כדי לשמור תחזיות, הקוד מאחזר אותן באמצעות השיטה `Results.to_df()` עבור כל פריים וכותב אותן בפורמט MOT-Challenge:
```txt
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```
בפורמט זה, שדה ה-`id` מוגדר כ-`-1` עבור זיהויים שעדיין לא Trail Track. עבור נתוני אמת טרקינג, ניתן, למשל, להגדיר את ערך ה-`conf` ל-1.
```txt
1,-1,1691.97,381.048,152.23,352.617,0.995616,-1,-1,-1
1,-1,1233.55,467.507,133.65,218.985,0.980069,-1,-1,-1
1,-1,108.484,461.531,97.759,297.453,0.942438,-1,-1,-1
1,-1,256.996,420.694,101.497,296.434,0.938051,-1,-1,-1
```
להלן דוגמה לפלט המתקבל משימוש בשיטה `to_df()` באובייקט תוצאות של Ultralytics:
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

שתי הפונקציות `save_mot_results` ו-`save_mot_from_json` יכולות גם לשמור תוויות פילוח (segmentation) בפורמט MOTS לפי בקשה. פורמט ה-MOTS מוגדר כדלקמן:
```txt
<frame> <id> <class_id> <img_height> <img_width> <rle>
```
בפורמט זה, `rle` מציין קידוד באורך ריצה כפי שבו משתמשים במערכת הנתונים COCO. לכן, ספריית `pycocotools` נדרשת לקידוד ופענוח מסכות בפורמט זה.

### שמירת תחזיות כ-JSON לשימוש עתידי
קובץ ה-JSON מכיל את התחזיות בפורמט JSON, שנוצרות באמצעות השיטה `Results.to_json()` עבור כל פריים.
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

## כיצד להשתמש ב-TrackEval
כדי להעריך תוצאות טרקינג, התחל על ידי שיבוט (clone) ריפוזיטורי המקורי [TrackEval](https://github.com/JonathonLuiten/TrackEval/):
```bash
git clone https://github.com/JonathonLuiten/TrackEval.git
```
לאחר מכן, העתק את תיקיית `data` שהוגדרה כהלכה — אשר אמורה להכיל את נתוני האמת והתחזיות לכל טרקר — אל תוך תיקיית TrackEval. כדי ליצור אוסף סינתטי של תחזיות טרקר ואמת, תוכל להשתמש בסקריפט הדוגמה `synthetic_dataset.py`.
כאשר יש לך את כל ההגדרות, הרץ את הסקריפט `run_mot_challenge.py` עם ארגומנטים מתאימים לשורת הפקודה כדי להעריך את מערכת הנתונים המותאמת שלך. סקריפט זה מחשב מדדי טרקינג כגון HOTA, MOTA, וכדומה. לפרטים נוספים, עיין בתיעוד של TrackEval.

**שימו לב:** הקפד להגדיר את `--DO_PREPROC` כ-`False` כדי להימנע מבעיות אפשריות. לדוגמה:
```bash
cd TrackEval
python scripts/run_mot_challenge.py --BENCHMARK MyCustomChallenge --SPLIT_TO_EVAL test --DO_PREPROC False
```

## פורמט הערכת MOT Challenge
**נתיב אמת טרקינג:**  
`/data/gt/mot_challenge/<YourChallenge>-<eval>/<SeqName>/gt/gt.txt`    
קובץ זה מכיל את תוויות אמת הטרקינג בפורמט MOT-Challenge. כאן, `<YourChallenge>` הוא שם האתגר או מערכת הנתונים שלך, `<eval>` הוא אחד מ-`train`, `test` או `all`, ו-`<SeqName>` מתייחס לרצף הווידאו.

מבנה הספריות תחת `/data/gt/mot_challenge/<YourChallenge>-<eval>` צריך להיות כדלקמן:
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

**דוגמה לקובץ `seqinfo.ini`:**
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

**קבצי סדרה:**  
בתוך תיקיית `/data/gt/mot_challenge/seqmaps`, צור קבצי טקסט המכילים את שמות הרצפים. לדוגמה, צור את הקבצים `<YourChallenge>-all.txt`, `<YourChallenge>-train.txt` ו-`<YourChallenge>-test.txt` עם המבנה הבא:
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

**נתיב התחזיות:**  
`/data/trackers/mot_challenge/<YourChallenge>-<eval>/<TrackerName>/data/<SeqName>.txt`   
קובץ זה מאחסן את התחזיות עבור כל רצף וידאו לכל טרקר. כאן, `<TrackerName>` הוא שם הטרקר או האלגוריתם בשימוש.

מבנה הספריות תחת `/data/trackers/mot_challenge/<YourChallenge>-<eval>` צריך להיות כדלקמן:
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
