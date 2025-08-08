import os
from utils import visualize, get_max_id, xml_to_mot
from synthetic_dataset import create_predictions, create_dataset, convert_dataset


def create_synthetic_dataset(
    videos_path: str = "dataset/source",
    gt_model: str = "models/salmons_filteredV1.pt",
    challenge_name: str = "MyCustomChallenge"
    ) -> None:
    """ Creates a synthetic tracking ground truth dataset for tracking
    using a YOLO model's predictions.

    :param str videos_path:
        Path to the folder where source videos are stored.
    :param str gt_model: 
        Path to the YOLO model's weigths to be used for ground truth.
    :param str challenge_name:
        Name to give at the challenge.
    """
    create_dataset(videos_path, gt_model, challenge_name)


def track_with_models(
        videos_path: str = "dataset/source",
        pred_models_path: str = "models",
        trackers: list = ["botsort", "bytetrack"],
        challenge_name: str = "TrackingSalmones",
        segment: bool = True,
        save_segmentation: bool = True
    ) -> None:

    pred_models = [os.path.join(pred_models_path, model) for model in os.listdir(pred_models_path)]
    create_predictions(videos_path, pred_models, trackers, challenge_name, segment=segment, save_segmentation=save_segmentation)


if __name__ == "__main__":
    # ? (Optional) Create the synthetic Dataset.
    # create_synthetic_dataset()

    # ? Convert a CVAT dataset to MOTchallenge format.
    #convert_dataset(cvat_labels="dataset/cvat_labels",
    #                challenge_name="TrackingSalmones")

    # ? Create predictions.
    # Create predictions results using other YOLO models and trackers.
    track_with_models()

    # ? Vizualice
    #videos_path = "dataset/source"
    #tracking_predictions_folder = "data/trackers/mot_challenge/MyCustomChallenge-test"
    #visualization_folder = "visualization"
    #diccionario_tracks = {}
#
    #for video in os.listdir(videos_path):
    #    video_path = os.path.join(videos_path, video)
    #    sequence_name = os.path.splitext(os.path.basename(video))[0] if os.path.isfile(video_path) else video
#
    #    for case in os.listdir(tracking_predictions_folder):
    #        case_data_folder = os.path.join(tracking_predictions_folder, case, "data")
    #        detections_file = os.path.join(case_data_folder, f"{sequence_name}.txt")
    #        segmentations_file = os.path.join(case_data_folder, f"{sequence_name}_seg.txt")
#
    #        visualization_path = os.path.join(visualization_folder, sequence_name, case)
    #        visualize(video_path=video_path, detections_path=detections_file, segmentations_path=segmentations_file,
    #                  output_path=visualization_path, use_detection=True, use_segmentation=True,
    #                  save_as_video=True, video_fps=15)
#
    #        diccionario_tracks[f"{case}_{sequence_name}"] = get_max_id(detections_file)
#
    #for key in sorted(diccionario_tracks.keys()):
    #    print(key, diccionario_tracks[key])