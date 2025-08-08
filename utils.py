import os
import json
from typing import Any
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET  # For reading CVAT labels
from pycocotools import mask as maskUtils  # For generating RLE
from ultralytics.engine.results import Results


def return_framerate(video_path: str) -> float:
    if not os.path.isfile(video_path):
        return 30.0

    if os.path.splitext(video_path)[1].lower() not in {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".flv",
        ".wmv",
        ".webm",
    }:
        return 30.0

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    return fps if fps > 0 else 30.0


# For saving YOLO results
def save_yolo_results_as_json(
    results: list[Results],
    file_path="results.json",
    normalize=False,
    decimals=20
):
    """ Save YOLO results for all frames into a single JSON file.
    
    The function iterates over each frame's result, converts it to a
    Python object (by converting the JSON string returned by
    .to_json()), and stores it in a dictionary using the frame index
    (as an integer) as the key.
    
    :param list results: A list of YOLO result objects, one per frame.
    :param (str, optional) file_path: The file path (including name)
        where the JSON will be saved. Defaults to "results.json".
    :param (bool, optional) normalize: Whether to normalize the JSON
        output. Defaults to False.
    decimals (int, optional): Number of decimal places for numerical
        values. Defaults to 20.
    """
    all_results = {}

    # Process each frame's result.
    for frame_idx, result in enumerate(results):
        # Convert the result (a Results object) to a JSON string,
        # then parse that string into a Python object
        # (typically a list of detection dictionaries).
        json_str = result.to_json(normalize=normalize, decimals=decimals)
        frame_detections = json.loads(json_str)
        # Use the frame index as key (plain integer, not zero-padded)
        all_results[frame_idx] = frame_detections

    # Save the complete results dictionary to a JSON file.
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to {file_path}")


# Formating results in MOT format
def save_mot_from_detections(
    detections_per_frame: dict[int, list[dict[str, Any]]],
    out_dir: str,
    sequence_name: str,
    include_classes: int | list[int] | None = None,
    fixed_confidence: bool = False,
    save_segmentation: bool = False,
    use_tracked_ids: bool = False,
    img_height: int = -1,
    img_width: int = -1,
):
    """ Save detections in MOTChallenge format from a dictionary of
    detections per frame.

    :param dict detections_per_frame: Dictionary with 0-based frame
        indices as keys and lists of detection dictionaries as values.
    :param str out_dir: Output directory for the text files.
    :param str sequence_name: Base name for the output files.
    :param (int, list, optional) include_classes: Classes to include in
        the output.
    :param (bool, optional) fixed_confidence: If True, it will use 1.0
        as confidence
    :param (bool, optional) save_segmentation: If True, save
        segmentation results.
    :param (bool, optional) use_tracked_ids: If true, use track IDs if
        they are available.
    :param (int, optional) img_height: Image height for segmentation.
    :param (int, optional) img_width: Image width for segmentation.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Define file paths for detections and, if applicable, segmentations.
    detection_file_path = os.path.join(out_dir, f"{sequence_name}.txt")
    seg_file_path = os.path.join(out_dir, f"{sequence_name}_seg.txt")

    det_lines = []
    seg_lines = []

    # Process results frame by frame.
    for frame_idx in sorted(detections_per_frame.keys()):
        frame_number = frame_idx + 1  # MOTChallenge uses 1-based frame.
        predictions = detections_per_frame[frame_idx]

        # Iterate over each detection in the frame.
        for pred in predictions:
            # Filter detections based on include_classes, if provided.
            class_val = pred.get("class")
            if isinstance(include_classes, int):
                if class_val != include_classes:
                    continue
            elif isinstance(include_classes, list):
                if class_val not in include_classes:
                    continue

            # Get bounding box information;
            # expected keys in 'box': x1, y1, x2, y2.
            box = pred.get("box", {})
            x1 = box.get("x1", 0)
            y1 = box.get("y1", 0)
            x2 = box.get("x2", 0)
            y2 = box.get("y2", 0)
            bb_left = x1
            bb_top = y1
            bb_width = x2 - x1
            bb_height = y2 - y1

            # Determine the confidence value.
            conf = 1.0 if fixed_confidence else pred.get("confidence", 0)

            # Determine the track ID to use. If a 'track_id' column
            # exists and use_tracked_ids is True, use its value;
            # otherwise, default to -1.
            if use_tracked_ids and "track_id" in pred:
                track_val = int(pred["track_id"])
            else:
                track_val = -1

            # Build the detection line in MOTChallenge format:
            # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>,
            # <bb_height>, <conf>, -1, -1, -1
            det_line = (f"{frame_number}, {track_val}, {bb_left}, {bb_top}, "
                        f"{bb_width}, {bb_height}, {conf}, -1, -1, -1")
            det_lines.append(det_line)

            # Process segmentation if requested.
            if not save_segmentation:
                continue
            seg_data = pred.get("segments", None)
            rle_str = ""
            if isinstance(seg_data, dict):
                # Expect dicts to be lists under keys 'x' and 'y'.
                xs = seg_data.get("x", [])
                ys = seg_data.get("y", [])
                if len(xs) == len(ys) and len(xs) > 0:
                    # Convert to a list of alternating coordinates.
                    polygon = [coord for pair in zip(xs, ys) for coord in pair]
                    # Convert the polygon into RLE using pycocotools.
                    rles = maskUtils.frPyObjects(
                        [polygon], img_height, img_width)
                    rle = maskUtils.merge(rles)
                    # Decode the 'counts' if it is in bytes.
                    if isinstance(rle["counts"], bytes):
                        rle_str = rle["counts"].decode("ascii")
                    else:
                        rle_str = rle["counts"]

            # Build the segmentation line in MOTS format:
            # <frame> <id> <class> <img_height> <img_width> <rle>
            seg_line = (f"{frame_number} {track_val} {class_val} {img_height} "
                        f"{img_width} {rle_str}")
            seg_lines.append(seg_line)

    # Write detection results to file.
    with open(detection_file_path, "w") as f:
        for line in det_lines:
            f.write(line + "\n")
    print(f"Detection results saved to {detection_file_path}")

    # Write segmentation results to file if required.
    if save_segmentation:
        with open(seg_file_path, "w") as f:
            for line in seg_lines:
                f.write(line + "\n")
        print(f"Segmentation results saved to {seg_file_path}")


def save_mot_from_results(
    results: list[Results],
    out_dir: str,
    sequence_name: str,
    include_classes: int | list[int] | None = None,
    fixed_confidence: bool = False,
    save_segmentation: bool = False,
    use_tracked_ids: bool = False,
    img_height: int = -1,
    img_width: int = -1,
):
    """ Save YOLO Results list in MOTChallenge format.

    :param list[Results] results: List of YOLO Results objects.
    :param str out_dir: Output directory for the text files.
    :param str sequence_name: Base name for the output files.
    :param (int, list, optional) include_classes: Classes to include in
        the output.
    :param (bool, optional) fixed_confidence: If True, it will use 1.0
        as confidence
    :param (bool, optional) save_segmentation: If True, save
        segmentation results.
    :param (bool, optional) use_tracked_ids: If true, use track IDs if
        they are available.
    :param (int, optional) img_height: Image height for segmentation.
    :param (int, optional) img_width: Image width for segmentation.
    """
    detections_per_frame = {}
    for frame_idx, result in enumerate(results):
        result_df = result.to_df(normalize=False, decimals=20)
        result_list_of_dicts = result_df.to_dict(orient="records")
        detections_per_frame[frame_idx] = result_list_of_dicts

    save_mot_from_detections(
        detections_per_frame,
        out_dir,
        sequence_name,
        include_classes=include_classes,
        fixed_confidence=fixed_confidence,
        save_segmentation=save_segmentation,
        use_tracked_ids=use_tracked_ids,
        img_height=img_height,
        img_width=img_width,
    )


def save_mot_from_json(
    json_path: str,
    out_dir: str,
    sequence_name: str,
    include_classes: int | list[int] | None = None,
    fixed_confidence: bool = False,
    save_segmentation: bool = False,
    use_tracked_ids: bool = False,
    img_height: int = -1,
    img_width: int = -1,
):
    """
    Save YOLO predictions from JSON in MOTChallenge format.

    :param str json_path: Path where the results are saved as JSON.
    :param str out_dir: Output directory for the text files.
    :param str sequence_name: Base name for the output files.
    :param (int, list, optional) include_classes: Classes to include in
        the output.
    :param (bool, optional) fixed_confidence: If True, it will use 1.0
        as confidence
    :param (bool, optional) save_segmentation: If True, save
        segmentation results.
    :param (bool, optional) use_tracked_ids: If true, use track IDs if
        they are available.
    :param (int, optional) img_height: Image height for segmentation.
    :param (int, optional) img_width: Image width for segmentation.
    """
    with open(json_path, "r") as f:
        all_results = json.load(f)
    detections_per_frame = {
        int(key): value for key, value in all_results.items()
    }

    save_mot_from_detections(
        detections_per_frame,
        out_dir,
        sequence_name,
        include_classes=include_classes,
        fixed_confidence=fixed_confidence,
        save_segmentation=save_segmentation,
        use_tracked_ids=use_tracked_ids,
        img_height=img_height,
        img_width=img_width,
    )


def read_xml(xml_path: str) -> tuple[dict[int, list[dict[str, Any]]], dict]:
    """
    Parse a CVAT for video 1.1 XML file and return:
      - detections_per_frame: {frame_idx: [ {class, name, box, confidence, track_id}, ... ]}
      - meta: {"img_width": int, "img_height": int, "label_to_id": {label_name: class_id}}

    Detection dict format:
      {
        "class": int,                 # class id
        "name": str,                  # label name
        "box": {"x1": float, "y1": float, "x2": float, "y2": float},
        "confidence": float,          # 1.0 for ground truth
        "track_id": int               # CVAT track id
      }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Defaults
    img_width = -1
    img_height = -1
    size = 0

    # Parse meta/original_size if present
    meta_el = root.find("meta")
    if meta_el is not None:
        orig_size_el = meta_el.find("task/original_size")
        size_text = meta_el.findtext("task/size")
        if size_text is not None:
            try:
                size = int(size_text)
            except Exception:
                size = 0
        if orig_size_el is not None:
            w_text = orig_size_el.findtext("width")
            h_text = orig_size_el.findtext("height")
            if w_text is not None:
                try:
                    img_width = int(w_text)
                except Exception:
                    img_width = -1
            if h_text is not None:
                try:
                    img_height = int(h_text)
                except Exception:
                    img_height = -1

    # Build label -> class id mapping from meta labels order
    label_to_id: dict[str, int] = {}
    if meta_el is not None:
        labels_el = meta_el.find("task/labels")
        if labels_el is not None:
            names: list[str] = []
            for lbl in labels_el.findall("label"):
                nm = lbl.findtext("name")
                if nm is not None:
                    names.append(nm)
            label_to_id = {nm: i for i, nm in enumerate(names)}

    # Fallback: collect unique labels from tracks (sorted) if meta labels missing
    if not label_to_id:
        unique_names = set()
        for t in root.findall("track"):
            lb = t.get("label")
            if lb is not None:
                unique_names.add(lb)
        for i, nm in enumerate(sorted(unique_names)):
            label_to_id[nm] = i

    detections_per_frame: dict[int, list[dict[str, Any]]] = {}

    # Iterate tracks and boxes
    for track in root.findall("track"):
        label_name = track.get("label", "")
        class_id = label_to_id.get(label_name, 0)

        try:
            track_id = int(track.get("id", "-1"))
        except Exception:
            track_id = -1

        for box_el in track.findall("box"):
            # Skip boxes marked outside="1"
            outside_attr = box_el.get("outside", "0")
            try:
                is_outside = int(outside_attr) == 1
            except Exception:
                is_outside = False
            if is_outside:
                continue

            # Frame index (0-based)
            try:
                frame_idx = int(box_el.get("frame", "0"))
            except Exception:
                continue

            # Coordinates
            try:
                xtl = float(box_el.get("xtl", "0"))
                ytl = float(box_el.get("ytl", "0"))
                xbr = float(box_el.get("xbr", "0"))
                ybr = float(box_el.get("ybr", "0"))
            except Exception:
                continue

            det = {
                "class": class_id,
                "name": label_name,
                "box": {"x1": xtl, "y1": ytl, "x2": xbr, "y2": ybr},
                "confidence": 1.0,
                "track_id": track_id,
            }

            if frame_idx not in detections_per_frame:
                detections_per_frame[frame_idx] = []
            detections_per_frame[frame_idx].append(det)

    meta = {
        "img_width": img_width,
        "img_height": img_height,
        "label_to_id": label_to_id,
        "size": size
    }
    return detections_per_frame, meta


def xml_to_mot(
    xml_path: str,
    out_dir: str,
    sequence_name: str,
    include_classes=None,
    fixed_confidence=False,
    save_segmentation=False,
    use_tracked_ids=False,
    img_height: int = -1,
    img_width: int = -1,
) -> tuple[int, int, int]:
    """
    Read CVAT-for-video-1.1 XML and save in MOTChallenge format using
    save_mot_from_detections().
    
    :param str xml_path: Path where the ground truth are saved as XML.
    :param str out_dir: Output directory for the text files.
    :param str sequence_name: Base name for the output files.
    :param (int, list, optional) include_classes: Classes to include in
        the output.
    :param (bool, optional) fixed_confidence: If True, it will use 1.0
        as confidence
    :param (bool, optional) save_segmentation: If True, save
        segmentation results.
    :param (bool, optional) use_tracked_ids: If true, use track IDs if
        they are available.
    :param (int, optional) img_height: Image height for segmentation.
    :param (int, optional) img_width: Image width for segmentation.
    """
    detections_per_frame, meta = read_xml(xml_path)

    # If not provided, fill image size from XML meta (useful for segmentation saving)
    if img_width is None or img_width < 0:
        img_width = meta.get("img_width", -1)
    if img_height is None or img_height < 0:
        img_height = meta.get("img_height", -1)
    length = meta.get("size", 0)

    # Persist using your existing helper
    save_mot_from_detections(
        detections_per_frame=detections_per_frame,
        out_dir=out_dir,
        sequence_name=sequence_name,
        include_classes=include_classes,
        fixed_confidence=fixed_confidence,
        save_segmentation=save_segmentation,
        use_tracked_ids=use_tracked_ids,
        img_height=img_height,
        img_width=img_width,
    )
    
    return img_height, img_width, length

def create_seqinfo_ini(
    save_path,
    seq_name,
    seq_length,
    frame_rate=30.0,
    img_width=1920,
    img_height=1080,
    img_dir="img1",
    img_ext=".jpg",
):
    """
    Creates a seqinfo.ini file in the specified directory with sequence
    information.

    Parameters:
        save_path (str): Directory where the seqinfo.ini file will be saved.
        seq_name (str): Name of the sequence.
        seq_length (int): Total number of frames in the sequence.
        frame_rate (float, optional): Frame rate of the sequence. Defaults to 30.
        img_width (int, optional): Width of the images. Defaults to 1920.
        img_height (int, optional): Height of the images. Defaults to 1080.
        img_dir (str, optional): Name of the directory containing images. Defaults to "img1".
        img_ext (str, optional): Image file extension (including the dot). Defaults to ".jpg".

    The generated INI file will have the following content:

        [Sequence]
        name=<SeqName>
        imDir=img1
        frameRate=30
        seqLength=525
        imWidth=1920
        imHeight=1080
        imExt=.jpg

    Example:
        create_seqinfo_ini("output", "people_walking_1_200", 525)
        -> Creates the file "output/seqinfo.ini"
    """
    # Ensure the save directory exists.
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, "seqinfo.ini")

    # Create and write the seqinfo.ini file.
    with open(file_path, "w") as f:
        f.write("[Sequence]\n")
        f.write(f"name={seq_name}\n")
        f.write(f"imDir={img_dir}\n")
        f.write(f"frameRate={frame_rate}\n")
        f.write(f"seqLength={seq_length}\n")
        f.write(f"imWidth={img_width}\n")
        f.write(f"imHeight={img_height}\n")
        f.write(f"imExt={img_ext}\n")

    print(f"seqinfo.ini created at {file_path}")


def update_seqmaps(seqmaps_dir, challenge_name, split_to_eval, sequence_name):
    """
    Update the seqmaps directory by adding the given sequence name to the appropriate sequence files.

    The function updates two files:
      1. <challenge_name>-<split_to_eval>.txt (e.g. YourChallenge-train.txt)
      2. <challenge_name>-all.txt

    For each file, the following rules apply:
      - If the file does not exist, it is created and the first line is "name".
      - If the file exists, the function checks if the sequence name is already present.
      - If the sequence name is not present, it is appended on a new line.

    Parameters:
        seqmaps_dir (str): Directory where the seqmaps files are stored.
        challenge_name (str): The name of the challenge (e.g., "YourChallenge").
        split_to_eval (str): The split identifier ("train", "test", or "all").
        sequence_name (str): The sequence name to add (e.g., "seqName1").

    Example file content for <challenge_name>-train.txt:
        name
        seqName1
        seqName2
        ...
    """
    # Ensure the seqmaps directory exists.
    os.makedirs(seqmaps_dir, exist_ok=True)

    # Define file paths for the split file and the "all" file.
    split_file = os.path.join(seqmaps_dir, f"{challenge_name}-{split_to_eval}.txt")
    all_file = os.path.join(seqmaps_dir, f"{challenge_name}-all.txt")

    def update_file(file_path, seq_name):
        """
        Helper function that creates/updates the file at file_path.
        If the file doesn't exist, it is created with a header "name".
        Then, if the sequence name is not already present, it is appended.
        """
        if not os.path.exists(file_path):
            # Create the file with the header and add the sequence name.
            with open(file_path, "w") as f:
                f.write("name\n")
                f.write(f"{seq_name}\n")
            print(f"Created '{file_path}' and added sequence: {seq_name}")
        else:
            # Read existing lines and check if the sequence name is already present.
            with open(file_path, "r") as f:
                lines = [line.strip() for line in f.readlines()]
            if seq_name in lines:
                print(f"Sequence '{seq_name}' already exists in '{file_path}'.")
            else:
                # Append the sequence name.
                with open(file_path, "a") as f:
                    f.write(f"{seq_name}\n")
                print(f"Added sequence '{seq_name}' to '{file_path}'.")

    # Update both the split file and the "all" file.
    update_file(split_file, sequence_name)
    update_file(all_file, sequence_name)


# For visualization
def load_detections(
    labels_path: str
)-> dict[int, list[tuple[int, float, float, float, float, float]]]:
    frame_dets = {}
    with open(labels_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            try:
                frame_idx = int(float(parts[0]))
                track_id = int(float(parts[1]))
                l, t, w, h, conf = map(float, parts[2:7])
            except ValueError:
                continue
            frame_dets.setdefault(frame_idx, []).append((track_id, l, t, w, h, conf))
    return frame_dets


def load_segmentations(labels_path: str) -> dict[int, list[tuple[int, str]]]:
    frame_segs: dict[int, list[tuple[int, str]]] = {}
    with open(labels_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) < 6:
                continue
            try:
                frame_idx = int(parts[0])
                track_id = int(parts[1])
                rle = " ".join(parts[5:])
            except ValueError:
                continue
            frame_segs.setdefault(frame_idx, []).append((track_id, rle))
    return frame_segs


def find_frame_path(directory: str, frame_idx: int) -> str | None:
    base = f"{frame_idx:05d}"
    for ext in (".jpg", ".png", ".jpeg"):
        p = os.path.join(directory, base + ext)
        if os.path.exists(p):
            return p
    return None


def annotate_detections(
    img: np.ndarray,
    dets: list[tuple[int, float, float, float, float, float]],
    trails: dict[int, list[tuple[int, int]]],
    color=(0, 255, 0),
) -> np.ndarray:
    for track_id, l, t, w, h, conf in dets:
        x, y, w_i, h_i = map(int, (l, t, w, h))
        cv2.rectangle(img, (x, y), (x + w_i, y + h_i), color, 2)
        cv2.putText(
            img,
            f"ID{track_id}:{conf:.2f}",
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
        cx, cy = x + w_i // 2, y + h_i // 2
        trails.setdefault(track_id, []).append((cx, cy))
    # draw trails
    for tid, points in trails.items():
        for i in range(1, len(points)):
            cv2.line(img, points[i - 1], points[i], color, 1)
    return img


def overlay_segmentations(
    img: np.ndarray,
    segs: list[tuple[int, str]],
    trails: dict[int, list[tuple[int, int]]],
    alpha: float = 0.5,
    color_mask=(0, 0, 255),
    color_trail=(255, 0, 0),
) -> np.ndarray:
    if maskUtils is None:
        raise ImportError("pycocotools is required for segmentation overlay")
    h, w = img.shape[:2]
    overlay = img.copy()
    for track_id, rle in segs:
        try:
            decoded = maskUtils.decode({"size": [h, w], "counts": rle})
        except Exception:
            continue
        overlay[decoded > 0] = color_mask
        ys, xs = np.where(decoded > 0)
        if ys.size and xs.size:
            cx, cy = int(xs.mean()), int(ys.mean())
            trails.setdefault(track_id, []).append((cx, cy))
    result = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    for tid, points in trails.items():
        for i in range(1, len(points)):
            cv2.line(result, points[i - 1], points[i], color_trail, 1)
    return result


def prune_trails(
    trails: dict[int, list[tuple[int, int]]],
    active_ids: list[int]
):
    # remove trails for IDs no longer present
    for tid in list(trails.keys()):
        if tid not in active_ids:
            del trails[tid]


def process_frames_from_directory(
    video_path: str,
    frame_dets,
    frame_segs,
    use_detection,
    use_segmentation,
    save_dir
):
    os.makedirs(save_dir, exist_ok=True)
    trails_det: dict[int, list[tuple[int, int]]] = {}
    trails_seg: dict[int, list[tuple[int, int]]] = {}
    frame_indices = sorted(set(frame_dets.keys()) | set(frame_segs.keys()))
    for frame_idx in frame_indices:
        img_path = find_frame_path(video_path, frame_idx)
        if not img_path:
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        # prune trails
        active_det_ids = [t[0] for t in frame_dets.get(frame_idx, [])]
        active_seg_ids = [t[0] for t in frame_segs.get(frame_idx, [])]
        if use_detection:
            prune_trails(trails_det, active_det_ids)
        if use_segmentation:
            prune_trails(trails_seg, active_seg_ids)
        if use_detection and frame_idx in frame_dets:
            img = annotate_detections(img, frame_dets[frame_idx], trails_det)
        if use_segmentation and frame_idx in frame_segs:
            img = overlay_segmentations(img, frame_segs[frame_idx], trails_seg)
        cv2.imwrite(os.path.join(save_dir, f"{frame_idx:05d}.jpg"), img)


def process_frames_from_video(
    video_path: str,
    frame_dets,
    frame_segs,
    use_detection,
    use_segmentation,
    save_dir,
    save_video=False,
    output_video_path="output.mp4",
    fps=30,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    os.makedirs(save_dir, exist_ok=True)
    trails_det: dict[int, list[tuple[int, int]]] = {}
    trails_seg: dict[int, list[tuple[int, int]]] = {}
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = None
    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        active_det_ids = [t[0] for t in frame_dets.get(frame_idx, [])]
        active_seg_ids = [t[0] for t in frame_segs.get(frame_idx, [])]
        if use_detection:
            prune_trails(trails_det, active_det_ids)
        if use_segmentation:
            prune_trails(trails_seg, active_seg_ids)
        if use_detection and frame_idx in frame_dets:
            frame = annotate_detections(frame, frame_dets[frame_idx], trails_det)
        if use_segmentation and frame_idx in frame_segs:
            frame = overlay_segmentations(frame, frame_segs[frame_idx], trails_seg)
        if save_video:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
            writer.write(frame)
        else:
            cv2.imwrite(os.path.join(save_dir, f"{frame_idx:05d}.jpg"), frame)
        frame_idx += 1
    cap.release()
    if writer:
        writer.release()


def visualize(
    video_path: str,
    detections_path: str,
    segmentations_path: str,
    output_path: str = "output_vis",
    use_detection: bool = True,
    use_segmentation: bool = False,
    save_as_video: bool = False,
    video_fps: int = 30,
):
    """
    Combined visualization for MOT detections and segmentations with track IDs and trails.
    """
    frame_dets = load_detections(detections_path) if use_detection else {}
    frame_segs = load_segmentations(segmentations_path) if use_segmentation else {}
    if os.path.isdir(video_path):
        process_frames_from_directory(
            video_path,
            frame_dets,
            frame_segs,
            use_detection,
            use_segmentation,
            output_path,
        )
    else:
        process_frames_from_video(
            video_path,
            frame_dets,
            frame_segs,
            use_detection,
            use_segmentation,
            output_path,
            save_as_video,
            os.path.join(output_path, "out.mp4"),
            video_fps,
        )


def get_max_id(file_path):
    max_id = -1  # Initialize to -1 in case file is empty
    try:
        with open(file_path, "r") as file:
            for line in file:
                # Split line by comma and strip whitespace
                values = [val.strip() for val in line.split(",")]
                if len(values) >= 2:  # Ensure line has at least frame and id
                    try:
                        current_id = int(values[1])  # ID is second column
                        max_id = max(max_id, current_id)
                    except ValueError:
                        continue  # Skip lines where ID isn't a valid integer
        return max_id
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None
