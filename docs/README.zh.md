**重要提示：关于翻译**

下文使用了 AI（自动翻译）工具进行翻译。由于此过程可能包含错误或不准确之处，建议您查看英文或西班牙文原版，以确保信息的准确性。

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

该仓库提供了将 Ultralytics YOLO 预测结果转换为 MOT-Challenge 格式的代码。同时，它还包含一个脚本，用于生成合成的跟踪真实标注数据集及相应的跟踪结果，从而方便使用 TrackEval 进行快速测试。

## 工具函数说明

核心功能均在 `utils.py` 文件中实现。

你可以直接使用 `save_mot_results` 函数，将 YOLO 模型生成的预测结果保存为 MOT-Challenge 格式。或者，如果你更习惯处理 JSON 文件，也可以先调用 `save_results_as_json` 函数生成预测结果的 JSON 文件，再通过 `save_mot_from_json` 将其转换为 MOT-Challenge 格式。

### 直接保存预测结果为 MOT 格式

保存预测时，代码会对每一帧调用 `Results.to_df()` 方法获取预测数据，并按照 MOT-Challenge 格式写入，格式如下：
```txt
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```
其中，对于尚未进行跟踪的检测，`id` 字段将设为 `-1`。而对于真实标注数据，你可以例如将 `conf` 值固定为 `1`。

```txt
1,-1,1691.97,381.048,152.23,352.617,0.995616,-1,-1,-1
1,-1,1233.55,467.507,133.65,218.985,0.980069,-1,-1,-1
1,-1,108.484,461.531,97.759,297.453,0.942438,-1,-1,-1
1,-1,256.996,420.694,101.497,296.434,0.938051,-1,-1,-1
```

下面展示了使用 Ultralytics Results 对象中的 `to_df()` 方法得到的示例输出：

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

另外，`save_mot_results` 和 `save_mot_from_json` 这两个函数在需要时也支持将分割标签保存为 MOTS 格式。MOTS 格式定义如下：

```txt
<frame> <id> <class_id> <img_height> <img_width> <rle>
```

这里的 `rle` 指的是 COCO 数据集中常用的 run-length encoding（RLE）编码，因此需要依赖 pycocotools 库来进行 mask 的编码和解码。

### 将预测结果保存为 JSON 文件以备后续使用

JSON 文件中保存了每一帧通过 `Results.to_json()` 方法生成的预测结果，格式为标准 JSON 格式。

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

## 如何使用 TrackEval

要评估跟踪结果，首先需要克隆官方的 [TrackEval](https://github.com/JonathonLuiten/TrackEval/) 仓库：

```bash
git clone https://github.com/JonathonLuiten/TrackEval.git
```

接下来，将配置妥当的 `data` 目录复制到 TrackEval 目录中（该目录中应包含你的每个 tracker 对应的真实标注和预测结果）。如果你需要生成一组合成的 tracker 预测和真实标注数据，可以使用示例脚本 `synthetic_dataset.py`。

配置完成后，使用合适的命令行参数运行 `run_mot_challenge.py` 脚本来评估你自定义的数据集。该脚本会计算 HOTA、MOTA 等跟踪评估指标。更多详情请参阅 TrackEval 的文档。

**注意：** 为避免潜在问题，请确保将 `--DO_PREPROC` 参数设置为 `False`。例如：

```bash
cd TrackEval
python scripts/run_mot_challenge.py --BENCHMARK MyCustomChallenge --SPLIT_TO_EVAL test --DO_PREPROC False
```

## MOT Challenge 评估格式

**真实标注路径：**  
`/data/gt/mot_challenge/<YourChallenge>-<eval>/<SeqName>/gt/gt.txt`  
该文件以 MOT-Challenge 格式存储真实标注数据。这里 `<YourChallenge>` 表示你的挑战或数据集名称，`<eval>` 可为 `train`、`test` 或 `all`，而 `<SeqName>` 则代表视频序列名称。

在 `/data/gt/mot_challenge/<YourChallenge>-<eval>` 目录下，文件结构应如下所示：

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

**示例 `seqinfo.ini` 文件：**

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

**序列文件：**  
在 `/data/gt/mot_challenge/seqmaps` 文件夹内，创建列出序列名称的文本文件。例如，创建 `<YourChallenge>-all.txt`、`<YourChallenge>-train.txt` 和 `<YourChallenge>-test.txt`，文件内容如下：

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

**预测结果路径：**  
`/data/trackers/mot_challenge/<YourChallenge>-<eval>/<TrackerName>/data/<SeqName>.txt`  
该文件存储了每个 tracker 对每个视频序列的预测结果，其中 `<TrackerName>` 表示所使用的跟踪算法或 tracker 名称。

在 `/data/trackers/mot_challenge/<YourChallenge>-<eval>` 目录下，文件结构应如下所示：

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