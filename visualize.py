# This code is just to check if data saved in the MOTchallenge format correctly.
# We use the labels saved in MOTchallenge format to then visualice said labels in the corresponding images.
import os
from utils import visualize

def get_max_id(file_path):
    max_id = -1  # Initialize to -1 in case file is empty
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Split line by comma and strip whitespace
                values = [val.strip() for val in line.split(',')]
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

def visualize_tracks(videos_path, tracking_predictions_folder, visualization_folder):
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


def visualize_gt(videos_path, ground_truth_folder, visualization_folder):
    diccionario_tracks = {}
    
    for video in os.listdir(videos_path):
        video_path = os.path.join(videos_path, video)
        sequence_name = os.path.splitext(os.path.basename(video))[0] if os.path.isfile(video_path) else video
        print(sequence_name)
        
        #for case in os.listdir(ground_truth_folder):
        #    print(case)

if __name__ == "__main__":
    videos_path = "dataset/source"
    tracking_predictions_folder = "data/trackers/mot_challenge/TrackingSalmones-test"
    ground_truth_folder = "data/gt/mot_challenge/TrackingSalmones-test"
    visualization_folder = "visualization"
    #visualize_tracks(videos_path, tracking_predictions_folder, visualization_folder)
    
    visualize_gt(videos_path, ground_truth_folder, visualization_folder)
