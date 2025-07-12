import os
from utils import visualize, get_max_id
from synthetic_dataset import create_predictions

if __name__ == "__main__":
    # Create the synthetic Dataset with yo°lov10x for Ground Truth labels.
    videos_path = "dataset/source"
    gt_model = "models/salmons_filteredV1.pt"
    challenge_name = "MyCustomChallenge"
    #create_dataset(videos_path, gt_model, challenge_name)

    # Create predictions results using other YOLO models and trackers.
    #pred_models = ["yolov8l-seg", "yolov9t", "yolov10s", "yolo11x"]
    pred_models = [os.path.join("models", model) for model in os.listdir("models")]
    trackers = ["botsort", "bytetrack"]
    create_predictions(videos_path, pred_models, trackers, challenge_name, segment=True)
    
    # Visualizar
    videos_path = "dataset/source"
    tracking_predictions_folder = "data/trackers/mot_challenge/MyCustomChallenge-test"
    visualization_folder = "visualization"
    diccionario_tracks = {}

    for video in os.listdir(videos_path):
        video_path = os.path.join(videos_path, video)
        sequence_name = os.path.splitext(os.path.basename(video))[0] if os.path.isfile(video_path) else video

        for case in os.listdir(tracking_predictions_folder):
            case_data_folder = os.path.join(tracking_predictions_folder, case, "data")
            detections_file = os.path.join(case_data_folder, f"{sequence_name}.txt")
            segmentations_file = os.path.join(case_data_folder, f"{sequence_name}_seg.txt")

            visualization_path = os.path.join(visualization_folder, sequence_name, case)
            visualize(video_path=video_path, detections_path=detections_file, segmentations_path=segmentations_file,
                      output_path=visualization_path, use_detection=True, use_segmentation=True,
                      save_as_video=True, video_fps=15)

            diccionario_tracks[f"{case}_{sequence_name}"] = get_max_id(detections_file)

    for key in sorted(diccionario_tracks.keys()):
        print(key, diccionario_tracks[key])