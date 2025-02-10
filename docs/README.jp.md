**翻訳に関する重要な注意事項**

以下のテキストは AI（自動翻訳）ツールを使用して翻訳されています。この過程には誤りや不正確さが含まれる可能性があるため、情報の正確性を確保するには 英語またはスペイン語の原文を参照することをお勧めします。

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
このリポジトリには、Ultralytics YOLO の予測結果を MOT-Challenge 形式に変換するコードが含まれています。また、TrackEval を使用して迅速なテストを可能にするために、対応するトラッキング結果とともに合成トラッキンググラウンドトゥルースデータセットを生成するスクリプトも含まれています。

## ユーティリティ関数の説明
主要な機能は `utils.py` ファイルに実装されています。

YOLO モデルによって生成された予測結果を直接 MOT-Challenge 形式で保存するには、`save_mot_results` 関数を使用します。もしくは、JSON ファイルで作業することを好む場合は、まず `save_results_as_json` 関数を使用して予測結果の JSON ファイルを取得し、その後 `save_mot_from_json` を用いてその JSON を MOT-Challenge 形式に変換することができます。

### 予測結果を直接 MOT 形式で保存
予測結果を保存するには、コードは各フレームごとに `Results.to_df()` メソッドを使用して予測を取得し、それらを MOT-Challenge 形式で書き出します:

```txt
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```

この形式では、トラッキングされていない検出に対して `id` フィールドが `-1` に設定されます。グラウンドトゥルースデータの場合、例えば `conf` の値を `1` に設定することができます。

```txt
1,-1,1691.97,381.048,152.23,352.617,0.995616,-1,-1,-1
1,-1,1233.55,467.507,133.65,218.985,0.980069,-1,-1,-1
1,-1,108.484,461.531,97.759,297.453,0.942438,-1,-1,-1
1,-1,256.996,420.694,101.497,296.434,0.938051,-1,-1,-1
```

以下は、Ultralytics の `Results` オブジェクトの `to_df()` メソッドを使用した際の出力例です:

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

`save_mot_results` と `save_mot_from_json` は、必要に応じて MOTS 形式でセグメンテーションラベルも保存することができます。MOTS 形式は以下のように定義されています:

```txt
<frame> <id> <class_id> <img_height> <img_width> <rle>
```

この形式では、`rle` は COCO データセットで利用されるランレングスエンコーディングを指します。そのため、この形式でマスクをエンコードおよびデコードするには、pycocotools ライブラリが必要となります。

### 将来のために JSON として予測結果を保存
JSON ファイルは、各フレームに対して `Results.to_json()` メソッドを使用して生成された JSON 形式の予測結果を含みます。

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

## TrackEval の使い方
トラッキング結果を評価するには、まず元の [TrackEval](https://github.com/JonathonLuiten/TrackEval/) リポジトリをクローンします:

```bash
git clone https://github.com/JonathonLuiten/TrackEval.git
```

次に、正しく構成された `data` ディレクトリ（これには各トラッカーのグラウンドトゥルースデータと予測結果が含まれている必要があります）を TrackEval ディレクトリにコピーします。合成のトラッカー予測とグラウンドトゥルースのコレクションを作成するために、例として `synthetic_dataset.py` スクリプトを使用することができます。

準備が整ったら、適切なコマンドライン引数を指定して `run_mot_challenge.py` スクリプトを実行し、カスタムデータセットを評価します。このスクリプトは HOTA、MOTA などのトラッキングメトリクスを計算します。詳細については、TrackEval のドキュメントを参照してください。

**注意:** 潜在的な問題を避けるため、`--DO_PREPROC` を `False` に設定してください。例えば:

```bash
cd TrackEval
python scripts/run_mot_challenge.py --BENCHMARK MyCustomChallenge --SPLIT_TO_EVAL test --DO_PREPROC False
```

## MOT Challenge 評価フォーマット

**グラウンドトゥルースのパス:**  
`/data/gt/mot_challenge/<YourChallenge>-<eval>/<SeqName>/gt/gt.txt`  
このファイルには、MOT-Challenge 形式のグラウンドトゥルースラベルが含まれています。ここで、`<YourChallenge>` はチャレンジまたはデータセットの名称、`<eval>` は `train`、`test`、あるいは `all` のいずれか、`<SeqName>` はビデオシーケンスに対応します。

`/data/gt/mot_challenge/<YourChallenge>-<eval>` 以下のディレクトリ構造は以下のようになっている必要があります:

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

**例: `seqinfo.ini` ファイル:**

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

**シーケンスファイル:**  
`/data/gt/mot_challenge/seqmaps` フォルダ内に、シーケンス名をリストしたテキストファイルを作成してください。例えば、`<YourChallenge>-all.txt`、`<YourChallenge>-train.txt`、および `<YourChallenge>-test.txt` を以下の構造で作成します:

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

**予測結果のパス:**  
`/data/trackers/mot_challenge/<YourChallenge>-<eval>/<TrackerName>/data/<SeqName>.txt`  
このファイルは、各トラッカーごとの各ビデオシーケンスに対する予測結果を保存します。ここで、`<TrackerName>` は使用されたトラッカーまたはアルゴリズムの名称です。

`/data/trackers/mot_challenge/<YourChallenge>-<eval>` 以下のディレクトリ構造は以下のようになっている必要があります:

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