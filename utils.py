import os
import json
import cv2
import numpy as np
from pycocotools import mask as maskUtils  # For generating RLE
import pandas as pd


# For saving YOLO results

def save_results_as_json(results, file_path="results.json", normalize=False, decimals=5):
    """
    Save YOLO results for all frames into a single JSON file.
    
    Parameters:
        results (list): A list of YOLO result objects, one per frame.
                        Each result must support the .to_json() method.
        file_path (str, optional): The file path (including name) where the JSON will be saved.
                                   Defaults to "results.json".
        normalize (bool, optional): Whether to normalize the JSON output.
                                    Passed to .to_json(). Defaults to False.
        decimals (int, optional): Number of decimal places for numerical values.
                                  Passed to .to_json(). Defaults to 5.
    
    The function iterates over each frame's result, converts it to a Python object
    (by converting the JSON string returned by .to_json()), and stores it in a dictionary
    using the frame index (as an integer) as the key.
    
    Example usage:
        results = model.predict(source=video_path)  # or model.track(source=video_path)
        save_results_as_json(results, file_path="my_results.json")
    """
    all_results = {}
    
    # Process each frame's result.
    for frame_idx, result in enumerate(results):
        # Convert the result (a Results object) to a JSON string,
        # then parse that string into a Python object (typically a list of detection dictionaries).
        json_str = result.to_json(normalize=normalize, decimals=decimals)
        frame_detections = json.loads(json_str)
        # Use the frame index as key (plain integer, not zero-padded)
        all_results[frame_idx] = frame_detections
    
    # Save the complete results dictionary to a JSON file with pretty indentation.
    with open(file_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to {file_path}")


# Formating results in MOT format

def save_mot_results(results, out_dir, sequence_name,
                     include_classes=None,
                     fixed_confidence=False,
                     save_segmentation=False,
                     use_tracked_ids=False,
                     img_height=-1, img_width=-1):
    """
    Save YOLO detection (and optional segmentation) results in MOTChallenge format.
    
    Parameters:
      results (list): List of YOLO result objects (each should support .to_df()).
      out_dir (str): Directory where the output .txt files will be saved.
      sequence_name (str): Base name of the sequence (used to name the files exactly).
      include_classes (int or list, optional): If provided, only detections with class(es)
                                               matching this value(s) will be saved.
      fixed_confidence (bool, optional): If True, sets each detection's confidence to 1.0;
                                         otherwise uses the detection's original confidence.
      save_segmentation (bool, optional): If True, an additional file with segmentation results
                                          (in MOTS Challenge format) will be saved.
      use_tracked_ids (bool, optional): If True and the detection results contain a 'track_id'
                                        column, the corresponding track IDs will be used in the output.
                                        If False or if no 'track_id' column is present, the track ID
                                        field will be set to -1.
      img_height (int, optional): Image height used for segmentation conversion.
                                  (Only used if save_segmentation is True and segmentation polygons
                                   are converted via pycocotools.)
      img_width (int, optional): Image width used for segmentation conversion.
                                 (Only used if save_segmentation is True.)
    
    MOTChallenge Format for detections:
      <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
      - Frame numbers are 1-based.
      - id is the track ID if use_tracked_ids is True and available; otherwise, it is -1.
      - x, y, z are set to -1.
    
    MOTS Challenge Format for segmentations:
      <frame> <id> <class_id> <img_height> <img_width> <rle>
      - 'rle' is the run-length encoding of the segmentation mask.
      - The provided image dimensions (img_height, img_width) are used for decoding the mask.
    
    The function processes each frame's results by converting the detections DataFrame (obtained via .to_df())
    into MOTChallenge lines. If segmentation saving is enabled, it converts the segmentation polygon into an
    RLE string using pycocotools.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Define file paths for detections and, if applicable, segmentations.
    detection_file_path = os.path.join(out_dir, f"{sequence_name}.txt")
    seg_file_path = os.path.join(out_dir, f"{sequence_name}_seg.txt")
    
    det_lines = []
    seg_lines = []
    
    # Process results frame by frame.
    for frame_idx, result in enumerate(results):
        frame_number = frame_idx + 1  # MOTChallenge uses 1-based frame numbering.
        df = result.to_df()  # Convert result to pandas DataFrame.
        
        # Iterate over each detection in the frame.
        for _, row in df.iterrows():
            class_val = row['class']
            # Filter detections based on include_classes if provided.
            if include_classes is not None:
                if isinstance(include_classes, int) and class_val != include_classes:
                    continue
                elif isinstance(include_classes, list) and class_val not in include_classes:
                    continue
            
            # Get bounding box information; expected keys in 'box': x1, y1, x2, y2.
            box = row['box']
            x1 = box.get('x1', 0)
            y1 = box.get('y1', 0)
            x2 = box.get('x2', 0)
            y2 = box.get('y2', 0)
            bb_left  = x1
            bb_top   = y1
            bb_width  = x2 - x1
            bb_height = y2 - y1
            
            # Determine the confidence value.
            conf = 1.0 if fixed_confidence else row['confidence']
            
            # Determine the track ID to use. If a 'track_id' column exists and use_tracked_ids is True,
            # use its value; otherwise, default to -1.
            if 'track_id' in row and use_tracked_ids:
                track_val = int(row['track_id'])
            else:
                track_val = -1
            
            # Build the detection line in MOTChallenge format.
            # Format: frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
            det_line = (f"{frame_number}, {track_val}, {bb_left:.2f}, {bb_top:.2f}, "
                        f"{bb_width:.2f}, {bb_height:.2f}, {conf:.5f}, -1, -1, -1")
            det_lines.append(det_line)
            
            # Process segmentation if requested.
            if save_segmentation:
                seg_data = row.get('segments', None)
                rle_str = ""
                if seg_data is not None and isinstance(seg_data, dict):
                    # Expect segmentation dict to contain lists under keys 'x' and 'y'.
                    xs = seg_data.get('x', [])
                    ys = seg_data.get('y', [])
                    if len(xs) == len(ys) and len(xs) > 0:
                        # Convert to a flat list of alternating coordinates.
                        polygon = []
                        for x, y in zip(xs, ys):
                            polygon.extend([x, y])
                        # Convert polygon to RLE using pycocotools. frPyObjects expects a list of polygons.
                        rles = maskUtils.frPyObjects([polygon], img_height, img_width)
                        rle = maskUtils.merge(rles)
                        # The 'counts' field may be bytes; decode if necessary.
                        if isinstance(rle['counts'], bytes):
                            rle_str = rle['counts'].decode('ascii')
                        else:
                            rle_str = rle['counts']
                # Build the segmentation line in MOTS Challenge format:
                # frame, track_id, class, img_height, img_width, rle_str.
                seg_line = f"{frame_number} {track_val} {class_val} {img_height} {img_width} {rle_str}"
                seg_lines.append(seg_line)
    
    # Write detection results to file.
    with open(detection_file_path, 'w') as f:
        for line in det_lines:
            f.write(line + "\n")
    
    # Write segmentation results if requested.
    if save_segmentation:
        with open(seg_file_path, 'w') as f:
            for line in seg_lines:
                f.write(line + "\n")
    
    print(f"Saved detection results to {detection_file_path}")
    if save_segmentation:
        print(f"Saved segmentation results to {seg_file_path}")


def save_mot_from_json(json_path, out_dir, sequence_name,
                       include_classes=None,
                       fixed_confidence=False,
                       save_segmentation=False,
                       use_tracked_ids=False,
                       img_height=-1, img_width=-1):
    """
    Convert and save YOLO predictions (stored in a JSON file) into MOTChallenge format.
    
    Parameters:
        json_path (str): Path to the JSON file containing YOLO predictions.
                         The JSON file should be a dictionary where each key is a frame number
                         (as an integer or string representing an integer) and its value is a list
                         of prediction dictionaries.
        out_dir (str): Directory where the output text files will be saved.
        sequence_name (str): Base name for the output files (e.g. "people_walking_1_200").
        include_classes (int or list, optional): If provided, only predictions with class(es)
                                                 matching this value(s) will be included.
        fixed_confidence (bool, optional): If True, sets each prediction's confidence to 1.0;
                                           otherwise, the original confidence is used.
        save_segmentation (bool, optional): If True, an additional file with segmentation results
                                            (in MOTS Challenge format) will be saved.
        use_tracked_ids (bool, optional): If True and a prediction dictionary contains a 'track_id'
                                          key, that value is used; otherwise, the track id is set to -1.
        img_height (int, optional): Image height used for segmentation conversion.
                                    Required if save_segmentation is True.
        img_width (int, optional): Image width used for segmentation conversion.
                                   Required if save_segmentation is True.
    
    MOTChallenge detection format:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        - Frame numbers are 1-based.
        - <id> is the track id if available and use_tracked_ids is True, otherwise -1.
    
    MOTS segmentation format (if saving segmentation results):
        <frame> <id> <class> <img_height> <img_width> <rle>
        - The RLE is computed from the segmentation polygon provided under the 'segments' key.
    
    The function reads the JSON file, processes each frame’s predictions, and writes the results
    to one text file for detections and, if requested, one for segmentation.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Load the predictions from the JSON file.
    with open(json_path, "r") as f:
        all_results = json.load(f)
    
    det_lines = []
    seg_lines = []
    
    # Iterate over each frame's predictions.
    # The keys in the JSON represent frame numbers.
    for frame_key, predictions in all_results.items():
        try:
            # Convert the frame key to an integer and adjust to 1-based indexing.
            frame_number = int(frame_key) + 1
        except ValueError:
            continue
        
        for pred in predictions:
            class_val = pred.get("class")
            # Filter predictions based on include_classes if provided.
            if include_classes is not None:
                if isinstance(include_classes, int) and class_val != include_classes:
                    continue
                elif isinstance(include_classes, list) and class_val not in include_classes:
                    continue
            
            # Extract bounding box information.
            box = pred.get("box", {})
            x1 = box.get("x1", 0)
            y1 = box.get("y1", 0)
            x2 = box.get("x2", 0)
            y2 = box.get("y2", 0)
            bb_left  = x1
            bb_top   = y1
            bb_width  = x2 - x1
            bb_height = y2 - y1
            
            # Determine the confidence value.
            conf = 1.0 if fixed_confidence else pred.get("confidence", 0)
            
            # Determine the track ID to use.
            if use_tracked_ids and "track_id" in pred:
                track_val = int(pred["track_id"])
            else:
                track_val = -1
            
            # Build the detection line in MOTChallenge format.
            # Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
            det_line = (f"{frame_number}, {track_val}, {bb_left:.2f}, {bb_top:.2f}, "
                        f"{bb_width:.2f}, {bb_height:.2f}, {conf:.5f}, -1, -1, -1")
            det_lines.append(det_line)
            
            # Process segmentation if requested.
            if save_segmentation:
                seg_data = pred.get("segments", None)
                rle_str = ""
                if seg_data is not None and isinstance(seg_data, dict):
                    xs = seg_data.get("x", [])
                    ys = seg_data.get("y", [])
                    if len(xs) == len(ys) and len(xs) > 0:
                        # Convert lists of x and y coordinates into a flat polygon list.
                        polygon = []
                        for x, y in zip(xs, ys):
                            polygon.extend([x, y])
                        # Convert the polygon into RLE using pycocotools.
                        rles = maskUtils.frPyObjects([polygon], img_height, img_width)
                        rle = maskUtils.merge(rles)
                        # Decode the 'counts' if it is in bytes.
                        if isinstance(rle["counts"], bytes):
                            rle_str = rle["counts"].decode("ascii")
                        else:
                            rle_str = rle["counts"]
                # Build the segmentation line in MOTS format:
                # <frame> <id> <class> <img_height> <img_width> <rle>
                seg_line = f"{frame_number} {track_val} {class_val} {img_height} {img_width} {rle_str}"
                seg_lines.append(seg_line)
    
    # Write detection results to file.
    detection_file_path = os.path.join(out_dir, f"{sequence_name}.txt")
    with open(detection_file_path, "w") as f:
        for line in det_lines:
            f.write(line + "\n")
    
    # Write segmentation results to file if required.
    if save_segmentation:
        seg_file_path = os.path.join(out_dir, f"{sequence_name}_seg.txt")
        with open(seg_file_path, "w") as f:
            for line in seg_lines:
                f.write(line + "\n")
        print(f"Segmentation results saved to {seg_file_path}")
    
    print(f"Detection results saved to {detection_file_path}")


def create_seqinfo_ini(save_path, seq_name, seq_length, 
                         frame_rate=30, im_width=1920, im_height=1080, 
                         im_dir="img1", im_ext=".jpg"):
    """
    Creates a seqinfo.ini file in the specified directory with sequence information.

    Parameters:
        save_path (str): Directory where the seqinfo.ini file will be saved.
        seq_name (str): Name of the sequence.
        seq_length (int): Total number of frames in the sequence.
        frame_rate (int, optional): Frame rate of the sequence. Defaults to 30.
        im_width (int, optional): Width of the images. Defaults to 1920.
        im_height (int, optional): Height of the images. Defaults to 1080.
        im_dir (str, optional): Name of the directory containing images. Defaults to "img1".
        im_ext (str, optional): Image file extension (including the dot). Defaults to ".jpg".
    
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
        f.write(f"imDir={im_dir}\n")
        f.write(f"frameRate={frame_rate}\n")
        f.write(f"seqLength={seq_length}\n")
        f.write(f"imWidth={im_width}\n")
        f.write(f"imHeight={im_height}\n")
        f.write(f"imExt={im_ext}\n")
    
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

def draw_detections(video_path, labels_path, output_path="output_visualization"):
    """
    Visualize detection results on video frames and save the output images.

    Parameters:
        video_path (str): Path to the directory containing video frames.
                          Frames are expected to be named as 00000.jpg, 00001.png, etc.
        labels_path (str): Path to the text file with detection results in MOTChallenge format.
                           Each line should be formatted as:
                           <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        output_path (str): Directory where the visualized images will be saved.
                           Defaults to "output_visualization".

    The function:
      - Reads the detection results from the text file.
      - Groups detections by frame number.
      - Searches for the corresponding image in video_path (supports multiple extensions).
      - Draws bounding boxes and confidence scores on the image.
      - Saves the output image in the specified output directory.
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Read detection results from the file.
    with open(labels_path, "r") as f:
        lines = f.readlines()
    
    # Organize detections by frame number.
    frame_detections = {}
    for line in lines:
        parts = line.strip().split(", ")
        # Expected parts: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
        try:
            frame = int(float(parts[0]))
            bb_left = float(parts[2])
            bb_top = float(parts[3])
            bb_width = float(parts[4])
            bb_height = float(parts[5])
            conf = float(parts[6])
        except (IndexError, ValueError):
            continue
        frame_detections.setdefault(frame, []).append((bb_left, bb_top, bb_width, bb_height, conf))
    
    # Process each frame and draw the detections.
    for frame, detections in frame_detections.items():
        img_name = f"{frame:05d}"
        img_path = None
        
        # Search for the image file with supported extensions.
        for ext in [".jpg", ".png", ".jpeg"]:
            test_path = os.path.join(video_path, img_name + ext)
            if os.path.exists(test_path):
                img_path = test_path
                break
        if img_path is None:
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Draw each detection (bounding box and confidence score)
        for bb_left, bb_top, bb_width, bb_height, conf in detections:
            x = int(bb_left)
            y = int(bb_top)
            w = int(bb_width)
            h = int(bb_height)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"Conf: {conf:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the modified image.
        cv2.imwrite(os.path.join(output_path, f"{img_name}.jpg"), img)


def draw_segmentations(video_path, labels_path, output_path="output_visualization_seg"):
    """
    Visualize segmentation results on video frames and save the output images.

    Parameters:
        video_path (str): Path to the directory containing video frames.
                          Frames are expected to be named as 00000.jpg, 00001.png, etc.
        labels_path (str): Path to the text file with segmentation results in MOTS Challenge format.
                           Each line should be formatted as:
                           <frame> <id> <class_id> <img_height> <img_width> <rle>
                           Note: The image dimensions in the file are ignored, as the actual dimensions
                           are read from each image.
        output_path (str): Directory where the visualized images will be saved.
                           Defaults to "output_visualization_seg".

    The function:
      - Reads segmentation results from the text file.
      - Groups RLE strings by frame number.
      - Searches for the corresponding image in video_path (supports multiple extensions).
      - Reads the image dimensions from the image itself.
      - For each RLE, decodes the mask using pycocotools.
      - Combines the masks and overlays them (in red) on the image.
      - Saves the blended image in the specified output directory.
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Read segmentation results from the file.
    with open(labels_path, "r") as f:
        lines = f.readlines()
    
    # Organize segmentation RLE strings by frame number.
    frame_segmentations = {}
    for line in lines:
        parts = line.strip().split(" ")
        # Expected format: <frame> <id> <class_id> <img_height> <img_width> <rle...>
        if len(parts) < 6:
            continue
        try:
            frame = int(parts[0])
        except ValueError:
            continue
        # Rejoin the remaining parts to reconstruct the RLE string.
        rle = " ".join(parts[5:]).strip()
        frame_segmentations.setdefault(frame, []).append(rle)
    
    # Process each frame and overlay the segmentation masks.
    for frame, rles in frame_segmentations.items():
        img_name = f"{frame:05d}"
        img_path = None
        
        # Search for the image file with supported extensions.
        for ext in [".jpg", ".png", ".jpeg"]:
            test_path = os.path.join(video_path, img_name + ext)
            if os.path.exists(test_path):
                img_path = test_path
                break
        if img_path is None:
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Get the actual image dimensions.
        h, w = img.shape[:2]
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Decode and combine each RLE mask.
        for rle in rles:
            try:
                decoded_mask = maskUtils.decode({"size": [h, w], "counts": rle})
            except Exception as e:
                print(f"Error decoding RLE for frame {frame}: {e}")
                continue
            seg_mask = np.maximum(seg_mask, decoded_mask)
        
        # Create an overlay where segmentation areas are marked in red.
        overlay = img.copy()
        overlay[seg_mask > 0] = [0, 0, 255]
        # Blend the overlay with the original image.
        alpha = 0.5
        blended = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
        
        # Save the resulting image.
        cv2.imwrite(os.path.join(output_path, f"{img_name}.jpg"), blended)
