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
from utils import return_framerate, save_yolo_results_as_json, save_mot_from_json, save_mot_from_results, xml_to_mot, create_seqinfo_ini, update_seqmaps


def convert_dataset(
    cvat_labels: str,
    challenge_name: str,
    split_to_eval: str | None = "test"
) -> None:
    for cvat_file in os.listdir(cvat_labels):
        video_name, ext = os.path.splitext(cvat_file)
        cvat_path = os.path.join(cvat_labels, cvat_file)
        
        # /data/gt/mot_challenge/<YourChallenge>-<eval>/<SeqName>/gt/gt.txt
        gt_path = os.path.join(
            'data', 'gt', 'mot_challenge',
            f"{challenge_name}-{split_to_eval}", video_name, 'gt'
        )
        xml_to_mot(
            xml_path=cvat_path,
            out_dir=gt_path,
            sequence_name='gt',
            include_classes=0,
            fixed_confidence=True,
            save_segmentation=False,
            use_tracked_ids=True
        )

# split_to_eval valid values: 'train', 'test', 'all'
def create_dataset(videos_path, model_name, challenge_name, split_to_eval="test"):
    # Create ground truth predictions for each video (sequence)
    for video in os.listdir(videos_path):
        video_name, ext = os.path.splitext(video)
        # Instanciate the YOLO model
        model = YOLO(model_name)
        video_path = os.path.join(videos_path, video)    # The path to the video or the directory of images.
        results = model.track(source=video_path, tracker="bytetrack.yaml")  # We are using bytetrack as example.
        height, width = results[0].orig_shape

        # Save the results in the MOT Challenge format in the path where ground truths go.
        # /data/gt/mot_challenge/<YourChallenge>-<eval>/<SeqName>/gt/gt.txt
        gt_path = os.path.join('data', 'gt', 'mot_challenge', f"{challenge_name}-{split_to_eval}", video_name, 'gt')
        save_mot_from_results(
            results=results,
            out_dir=gt_path,
            sequence_name='gt',
            include_classes=0,
            fixed_confidence=True,
            save_segmentation=True,
            use_tracked_ids=True,
            img_height=height,
            img_width=width
        )

        # Create the seqinfo.ini file.
        seq_ini_path = os.path.join('data', 'gt', 'mot_challenge', f"{challenge_name}-{split_to_eval}", video_name)
        seq_length = len(results)
        frame_rate = return_framerate(video_path)
        create_seqinfo_ini(save_path=seq_ini_path, seq_name=video_name, seq_length=seq_length, frame_rate=frame_rate, im_width=width, im_height=height)

        del model   # Delete model to save memory


def create_predictions(videos_path, models_path, trackers_list, challenge_name, segment=False, save_segmentation=False, split_to_eval="test"):
    json_folder = "predictions_json"
    # Create predictions for each video (sequence)
    for model_path in models_path:
        for tracker in trackers_list:
            # Create predictions for each video (sequence)
            for video in os.listdir(videos_path):
                video_path = os.path.join(videos_path, video)    # The path to the video or the directory of images.
                model_name = os.path.splitext(os.path.basename(model_path))[0]
                pred_path = os.path.join('data', 'trackers', 'mot_challenge', f"{challenge_name}-{split_to_eval}", f"{model_name}_{tracker}", "data")
                sequence_name = os.path.splitext(os.path.basename(video))[0] if os.path.isfile(video_path) else video

                # Predict values
                model = YOLO(model_path, task="segment") if segment else YOLO(model_path)
                results = model.track(source=video_path, tracker=f"{tracker}.yaml")
                height, width = results[0].orig_shape
                # Save in Json
                json_file = os.path.join(json_folder, sequence_name, f"{model_name}_{tracker}.json")
                save_yolo_results_as_json(results=results, file_path=json_file, normalize=False)

                del model   # Delete model to save memory
                del results

                # Save the results in the MOT Challenge format in the path where predictions go.
                # /data/trackers/mot_challenge/<YourChallenge>-<eval>/<TrackerName>/data/<SeqName01>.txt
                save_mot_from_json(json_path=json_file, out_dir=pred_path, sequence_name=sequence_name, include_classes=0, fixed_confidence=False,
                                   save_segmentation=save_segmentation, use_tracked_ids=True, img_height=height, img_width=width)

                # Add it to the sequence map files.
                seqmaps_dir = os.path.join('data', 'gt', 'mot_challenge', 'seqmaps')
                update_seqmaps(seqmaps_dir=seqmaps_dir, challenge_name=challenge_name, split_to_eval=split_to_eval, sequence_name=sequence_name)


if __name__ == "__main__":
    # Create the synthetic Dataset with yolov10x for Ground Truth labels.
    videos_path = "dataset/source"
    gt_model = "models/salmons_filteredV1.pt"
    challenge_name = "MyCustomChallenge"
    #create_dataset(videos_path, gt_model, challenge_name)

    # Create predictions results using other YOLO models and trackers.
    #pred_models = ["yolov8l-seg", "yolov9t", "yolov10s", "yolo11x"]
    pred_models = [os.path.join("models", model) for model in os.listdir("models")]
    trackers = ["botsort", "bytetrack"]
    create_predictions(videos_path, pred_models, trackers, challenge_name, segment=True)

    # Now copy the "data" folder into the TrackEval directory.
    # Then run the next code in the console while being in the TrackEval folder:
    # python scripts/run_mot_challenge.py --BENCHMARK MyCustomChallenge --SPLIT_TO_EVAL test --DO_PREPROC False