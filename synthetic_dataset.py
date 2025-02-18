# The idea of this code is to create a synthetic tracking dataset to testing purposes.
# 1) Use YOLOv10x to track some videos.
# 2) Save the results of said predictions in the MOT Challenge format.
# 3) Use other YOLO models (any that is not the same used for Ground Truth) to make predictions
# 4) Save the predictions in the MOT Challenge format.
# 5) Export the files to the TrackEval proyect to test it with this new data.
# Then I can easily test the TrackEval code with custom data.

"""
Ground Truth Path: /data/gt/mot_challenge/<YourChallenge>-<eval>/<SeqName>/gt/gt.txt
This is the file containing the ground truths labels in MOT Challenge format. "eval" is either "test", "train" or "all".

.
|—— <SeqName01>
	|—— gt
		|—— gt.txt
	|—— seqinfo.ini
|—— <SeqName02>
	|—— ……
|—— <SeqName03>
	|—— …...

---
seqinfo.ini example

[Sequence]
name=<SeqName>
imDir=img1
frameRate=30
seqLength=525
imWidth=1920
imHeight=1080
imExt=.jpg

---
Sequence File: There is the folder "/data/gt/mot_challenge/seqmaps" that includes sequence files.
Create text files containing the sequence names; <YourChallenge>-train.txt, <YourChallenge>-test.txt, <YourChallenge>-test.txt inside the seqmaps folder, e.g.:

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


----
Predictions Path: /data/trackers/mot_challenge/<YourChallenge>-<eval>/<TrackerName>/data/<SeqName>.txt
"""
 

import os
from ultralytics import YOLO
from utils import save_mot_results, create_seqinfo_ini, update_seqmaps


# split_to_eval valid values: 'train', 'test', 'all'
def create_dataset(videos_path, model_name, challenge_name, split_to_eval="test"):
    # Create ground truth predictions for each video (sequence)
    for video in os.listdir(videos_path):
        # Instanciate the YOLO model
        model = YOLO(model_name)
        
        video_path = os.path.join(videos_path, video)    # The path to the video or the directory of images.
        results = model.track(source=video_path, tracker="bytetrack.yaml")  # We are using bytetrack as example.

        # Save the results in the MOT Challenge format in the path where ground truths go.
        # /data/gt/mot_challenge/<YourChallenge>-<eval>/<SeqName>/gt/gt.txt
        gt_path = os.path.join('data', 'gt', 'mot_challenge', f"{challenge_name}-{split_to_eval}", video, 'gt')
        save_mot_results(results=results, out_dir=gt_path, sequence_name='gt',
                         include_classes=0, fixed_confidence=True, save_segmentation=False,
                         use_tracked_ids=True)

        # Create the seqinfo.ini file.
        seq_ini_path = os.path.join('data', 'gt', 'mot_challenge', f"{challenge_name}-{split_to_eval}", video)
        seq_length = len(results)
        height, width = results[0].orig_shape
        create_seqinfo_ini(save_path=seq_ini_path, seq_name=video, seq_length=seq_length, im_width=width, im_height=height)
        
        del model   # Delete model to save memory


def create_predictions(videos_path, models_list, trackers_list, challenge_name, split_to_eval="test"):
    # Create predictions for each video (sequence)
    for model_name in models_list:
        for tracker in trackers_list:
            for video in os.listdir(videos_path):
                # Instanciate the YOLO model
                model = YOLO(model_name)
                
                video_path = os.path.join(videos_path, video)    # The path to the video or the directory of images.
                results = model.track(source=video_path, tracker=f"{tracker}.yaml")
                
                del model   # Delete model to save memory

                # Save the results in the MOT Challenge format in the path where predictions go.
                # /data/trackers/mot_challenge/<YourChallenge>-<eval>/<TrackerName>/data/<SeqName01>.txt
                pred_path = os.path.join('data', 'trackers', 'mot_challenge',
                                         f"{challenge_name}-{split_to_eval}",
                                         f"{model_name}_{tracker}", "data")
                save_mot_results(results=results, out_dir=pred_path, sequence_name=video,
                                include_classes=0, fixed_confidence=False, save_segmentation=False,
                                use_tracked_ids=True)

                # Add it to the sequence map files.
                seqmaps_dir = os.path.join('data', 'gt', 'mot_challenge', 'seqmaps')
                update_seqmaps(seqmaps_dir=seqmaps_dir, challenge_name=challenge_name,
                               split_to_eval=split_to_eval, sequence_name=video)


if __name__ == "__main__":
    # Create the synthetic Dataset with yolov10x for Ground Truth labels.
    videos_path = "dataset/source"
    gt_model = "yolov10x"
    challenge_name = "MyCustomChallenge"
    # create_dataset(videos_path, gt_model, challenge_name)
    
    # Create predictions results using other YOLO models and trackers.
    pred_models = ["yolov8l-seg", "yolov9t", "yolov10s", "yolo11x"]
    trackers = ["botsort", "bytetrack"]
    # create_predictions(videos_path, pred_models, trackers, challenge_name)
    
    # Now copy the "data" folder into the TrackEval directory.
    # Then run the next code in the console while being in the TrackEval folder:
    # python scripts/run_mot_challenge.py --BENCHMARK MyCustomChallenge --SPLIT_TO_EVAL test --DO_PREPROC False