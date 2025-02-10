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

This repository contains code to convert Ultralytics YOLO predictions into the MOT-Challenge format. It also includes a script to generate a synthetic tracking ground truth dataset along with corresponding tracking results, enabling quick testing with TrackEval.

## Utility Functions Explanation

The core functionality is implemented in the `utils.py` file.

You can save predictions produced by a YOLO model directly in MOT-Challenge format using the `save_mot_results` function. Alternatively, if you prefer to work with JSON files, you can first obtain a JSON file of predictions using the `save_results_as_json` function, and then convert that JSON into MOT-Challenge format with `save_mot_from_json`.

### Save Predictions Directly to MOT Format

To save predictions, the code retrieves them using the `Results.to_df()` method for each frame and writes them in the MOT-Challenge format:
```txt
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```
In this format, the `id` field is set to `-1` for detections that have not been tracked yet. For ground truth data, you can, for example, set the `conf` value to `1`.

```txt
1,-1,1691.97,381.048,152.23,352.617,0.995616,-1,-1,-1
1,-1,1233.55,467.507,133.65,218.985,0.980069,-1,-1,-1
1,-1,108.484,461.531,97.759,297.453,0.942438,-1,-1,-1
1,-1,256.996,420.694,101.497,296.434,0.938051,-1,-1,-1
```

Below is an example output from using the `to_df()` method in an Ultralytics Results object:

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

Both `save_mot_results` and `save_mot_from_json` can also save segmentation labels in the MOTS format if requested. The MOTS format is defined as follows:

```txt
<frame> <id> <class_id> <img_height> <img_width> <rle>
```

In this format, `rle` stands for run-length encoding as used in the COCO dataset. Consequently, the pycocotools library is required to encode and decode masks in this format.

### Save Predictions as JSON for Future Use

The JSON file contains the predictions in JSON format, generated using the `Results.to_json()` method for each frame.

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

## How to Use TrackEval

To evaluate tracking results, start by cloning the original [TrackEval](https://github.com/JonathonLuiten/TrackEval/) repository:

```bash
git clone https://github.com/JonathonLuiten/TrackEval.git
```

Next, copy the correctly configured `data` directory —which should contain your ground truth data and predictions for each tracker—into the TrackEval directory. To create a synthetic collection of tracker predictions and ground truths, you can use the example script `synthetic_dataset.py`.

Once you have everything set up, run the `run_mot_challenge.py` script with the appropriate command line arguments to evaluate your custom dataset. This script computes tracking metrics such as HOTA, MOTA, etc. For more details, refer to the TrackEval documentation.

**Note:** Ensure you set `--DO_PREPROC` to `False` to avoid potential issues. For example:

```bash
cd TrackEval
python scripts/run_mot_challenge.py --BENCHMARK MyCustomChallenge --SPLIT_TO_EVAL test --DO_PREPROC False
```

## MOT Challenge Evaluation Format

**Ground Truth Path:**  
`/data/gt/mot_challenge/<YourChallenge>-<eval>/<SeqName>/gt/gt.txt`     
This file contains the ground truth labels in MOT-Challenge format. Here, `<YourChallenge>` is the name of your challenge or dataset, `<eval>` is either `train`, `test`, or `all`, and `<SeqName>` corresponds to the video sequence.

The directory structure under `/data/gt/mot_challenge/<YourChallenge>-<eval>` should be as follows:

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

**Example `seqinfo.ini` file:**

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

**Sequence Files:**  
Within the `/data/gt/mot_challenge/seqmaps` folder, create text files listing the sequence names. For example, create `<YourChallenge>-all.txt`, `<YourChallenge>-train.txt`, and `<YourChallenge>-test.txt` with the following structure:

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

**Predictions Path:**  
`/data/trackers/mot_challenge/<YourChallenge>-<eval>/<TrackerName>/data/<SeqName>.txt`  
This file stores the predictions for each video sequence for each tracker. Here, `<TrackerName>` is the name of the tracker or algorithm used.

The directory structure under `/data/trackers/mot_challenge/<YourChallenge>-<eval>` should be as follows:

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
