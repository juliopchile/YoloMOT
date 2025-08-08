import os
from TrackEval import trackeval

CODE_PATH = ""

DEFAULT_EVAL_CONFIG = {
    'USE_PARALLEL': False,
    'NUM_PARALLEL_CORES': 8,
    'BREAK_ON_ERROR': True,  # Raises exception and exits with error
    'RETURN_ON_ERROR': False,  # if not BREAK_ON_ERROR, then returns from function on error
    'LOG_ON_ERROR': os.path.join(CODE_PATH, 'error_log.txt'),  # if not None, save any errors into a log file.
    'PRINT_RESULTS': True,
    'PRINT_ONLY_COMBINED': False,
    'PRINT_CONFIG': True,
    'TIME_PROGRESS': True,
    'DISPLAY_LESS_PROGRESS': True,
    'OUTPUT_SUMMARY': True,
    'OUTPUT_EMPTY_CLASSES': True,  # If False, summary files are not output for classes with no detections
    'OUTPUT_DETAILED': True,
    'PLOT_CURVES': True,
}

DEFAULT_DATASET_CONFIG = {
    'GT_FOLDER': os.path.join(CODE_PATH, 'data/gt/mot_challenge/'),  # Location of GT data
    'TRACKERS_FOLDER': os.path.join(CODE_PATH, 'data/trackers/mot_challenge/'),  # Trackers location
    'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
    'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
    'CLASSES_TO_EVAL': ['pedestrian'],  # Valid: ['pedestrian']
    'BENCHMARK': 'MOT17',  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
    'SPLIT_TO_EVAL': 'train',  # Valid: 'train', 'test', 'all'
    'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
    'PRINT_CONFIG': True,  # Whether to print current config
    'DO_PREPROC': True,  # Whether to perform preprocessing (never done for MOT15)
    'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
    'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
    'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
    'SEQMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
    'SEQMAP_FILE': None,  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
    'SEQ_INFO': None,  # If not None, directly specify sequences to eval and their number of timesteps
    'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',  # '{gt_folder}/{seq}/gt/gt.txt'
    'SKIP_SPLIT_FOL': False,  # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
                                # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
                                # If True, then the middle 'benchmark-split' folder is skipped for both.
}


def clean_trackings(folder="TrackEval/data/trackers/mot_challenge/TrackingSalmones-test", dry_run=True):
    """
    Clean tracking files using only 'os'.
    - Remove rows where id == -1
    - Remap duplicate IDs appearing in the same frame to new unique integers
    - Overwrite files directly (no backup)
    Args:
        folder: root folder containing per-tracker subfolders with a 'data' subfolder and .txt files.
        dry_run: if True, do not write files (preview only)
    """
    trackers_directory = os.path.abspath(folder)
    if not os.path.isdir(trackers_directory):
        print("Folder not found:", trackers_directory)
        return

    for tracker in os.listdir(trackers_directory):
        tracker_dir = os.path.join(trackers_directory, tracker)
        tracker_folder = os.path.join(tracker_dir, "data")
        if not os.path.isdir(tracker_folder):
            continue

        for filename in os.listdir(tracker_folder):
            if not filename.endswith(".txt"):
                continue

            file_path = os.path.join(tracker_folder, filename)
            print("\nProcessing:", file_path)

            # read file content
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_lines = f.read().splitlines()
            except Exception as e:
                print("  could not read file:", e)
                continue

            # first pass: parse and find max id
            parsed = []  # list of tuples (is_data_bool, tokens_or_raw)
            max_id = -1
            for line in raw_lines:
                s = line.strip()
                if s == "" or s.startswith("#"):
                    parsed.append((False, line))
                    continue

                if filename.endswith("_seg.txt"):
                    parts = s.split()
                    if len(parts) < 2:
                        parsed.append((False, line))
                        continue
                    try:
                        id_val = int(float(parts[1]))
                    except Exception:
                        parsed.append((False, line))
                        continue
                    parsed.append((True, parts))
                    if id_val > max_id:
                        max_id = id_val
                else:
                    parts = [p.strip() for p in s.split(",")]
                    if len(parts) < 2:
                        parsed.append((False, line))
                        continue
                    try:
                        id_val = int(float(parts[1]))
                    except Exception:
                        parsed.append((False, line))
                        continue
                    parsed.append((True, parts))
                    if id_val > max_id:
                        max_id = id_val

            # second pass: remove -1 ids and remap duplicates per-frame
            next_id = max_id + 1
            seen = {}  # frame -> set(ids seen)
            removed_count = 0
            remapped_count = 0
            out_lines = []

            for is_data, tokens in parsed:
                if not is_data:
                    out_lines.append(tokens)
                    continue

                if filename.endswith("_seg.txt"):
                    parts = tokens[:]  # list of strings
                    try:
                        frame = int(float(parts[0]))
                        id_val = int(float(parts[1]))
                    except Exception:
                        out_lines.append(" ".join(parts))
                        continue

                    if id_val == -1:
                        removed_count += 1
                        continue

                    if frame not in seen:
                        seen[frame] = set()
                    if id_val in seen[frame]:
                        new_id = next_id
                        next_id += 1
                        remapped_count += 1
                        parts[1] = str(new_id)
                        seen[frame].add(new_id)
                    else:
                        seen[frame].add(id_val)

                    out_lines.append(" ".join(parts))
                else:
                    parts = tokens[:]  # CSV tokens
                    try:
                        frame = int(float(parts[0]))
                        id_val = int(float(parts[1]))
                    except Exception:
                        out_lines.append(", ".join(parts))
                        continue

                    if id_val == -1:
                        removed_count += 1
                        continue

                    if frame not in seen:
                        seen[frame] = set()
                    if id_val in seen[frame]:
                        new_id = next_id
                        next_id += 1
                        remapped_count += 1
                        parts[1] = str(new_id)
                        seen[frame].add(new_id)
                    else:
                        seen[frame].add(id_val)

                    out_lines.append(", ".join(parts))

            print("  removed rows with id==-1:", removed_count)
            print("  remapped duplicates:", remapped_count)

            if dry_run:
                print("  dry_run=True -> not writing changes.")
                continue

            # overwrite file directly (no backup)
            try:
                with open(file_path, "w", encoding="utf-8") as fw:
                    if out_lines:
                        fw.write("\n".join(out_lines) + "\n")
                    else:
                        fw.write("")  # empty file
                print("  written cleaned file (no backup).")
            except Exception as e:
                print("  write failed:", e)



def run_my_eval(challenge_name: str = "TrackingSalmones",
                split_to_eval: str = "test"):
    output_folder = os.path.abspath("TrackEvalResults")

    # 1) obtener configuraciones por defecto
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_eval_config['USE_PARALLEL'] = True
    default_eval_config['OUTPUT_EMPTY_CLASSES'] = False
    default_eval_config['PLOT_CURVES'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}

    # 2) merge con tus overrides
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}

    # Overrides que quieres aplicar
    overrides = {
        'PRINT_CONFIG': False,
        'BENCHMARK': challenge_name,
        'SPLIT_TO_EVAL': split_to_eval,
        'DO_PREPROC': False,
        'OUTPUT_FOLDER': output_folder
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
    # This code requires an older version of numpy. Change enviromnet
    # if necesary.
    #clean_trackings(dry_run=False)
    run_my_eval()

    # ? Or try
    #* cd TrackEval
    #* python scripts/run_mot_challenge.py --BENCHMARK TrackingSalmones --SPLIT_TO_EVAL test --DO_PREPROC False