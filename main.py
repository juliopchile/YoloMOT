import os
import sys
from utils import visualize, get_max_id, xml_to_mot
from synthetic_dataset import make_predictions, create_dataset, convert_dataset
from TrackEval import trackeval


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
        save_segmentation: bool = False
    ) -> None:

    pred_models = [os.path.join(pred_models_path, model) for model in os.listdir(pred_models_path)]
    make_predictions(videos_path, pred_models, trackers, challenge_name, segment=segment, save_segmentation=save_segmentation)


def vizualice():
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


def run_my_eval(challenge_name: str = "TrackingSalmones",
                split_to_eval: str = "test"):
    # 1) obtener configuraciones por defecto
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}

    # 2) merge con tus overrides
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}

    # Overrides que quieres aplicar
    overrides = {
        'BENCHMARK': challenge_name,
        'SPLIT_TO_EVAL': split_to_eval,
        'DO_PREPROC': False,
    }
    config.update(overrides)

    # 3) separar las sub-configs como lo hace el script original
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # 4) crear evaluator, dataset y métricas
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)

    metrics_list = []
    for metric_class in [trackeval.metrics.HOTA,
                         trackeval.metrics.CLEAR,
                         trackeval.metrics.Identity,
                         trackeval.metrics.VACE]:
        # el script original usa metric_class.get_name() sin instanciar
        if metric_class.get_name() in metrics_config['METRICS']:
            # instanciar la métrica con metrics_config
            metrics_list.append(metric_class(metrics_config))

    if not metrics_list:
        raise Exception('No metrics selected for evaluation')

    # 5) ejecutar evaluación
    evaluator.evaluate([dataset], metrics_list)


if __name__ == "__main__":
    # ? (Optional) Create the synthetic Dataset.
    # create_synthetic_dataset()

    # ? Convert a CVAT dataset to MOTchallenge format.
    convert_dataset(cvat_labels="dataset/cvat_labels",
                    videos_path="dataset/source",
                    challenge_name="TrackingSalmones")

    # ? Create predictions.
    # Create predictions results using other YOLO models and trackers.
    track_with_models()

    # ? Vizualice
    vizualice()

    # ? TrackEval
    # run_my_eval()
    #! Better try using the run_track_eval.py file

    # ? Or try
    #* cd TrackEval
    #* python scripts/run_mot_challenge.py --BENCHMARK TrackingSalmones --SPLIT_TO_EVAL test --DO_PREPROC False
