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


if __name__ == "__main__":
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

# SalmonesV1_botsort 341, 225
# SalmonesV1_bytetrack 305, 279
# SalmonesV1int8_botsort 197, 169
# SalmonesV1int8_bytetrack 179, 202
# SalmonesV11_botsort 198, 170
# SalmonesV11_bytetrack 187, 196
# SalmonesV11int8_botsort 134, 116
# SalmonesV11int8_bytetrack 149, 131

# SalmonesV2_botsort 93, 153
# SalmonesV2_bytetrack 96, 174
# SalmonesV2int8_botsort 46, 86
# SalmonesV2int8_bytetrack 57, 102
# SalmonesV12_botsort 87, 146
# SalmonesV12_bytetrack 79, 168
# SalmonesV12int8_botsort 53, 92
# SalmonesV12int8_bytetrack 61, 104
# SalmonesV22_botsort 83, 129
# SalmonesV22_bytetrack 80, 155
# SalmonesV22int8_botsort 60, 106
# SalmonesV22int8_bytetrack 68, 112

# SalmonesV3_botsort 103, 170
# SalmonesV3_bytetrack 110, 183
# SalmonesV3int8_botsort 66, 99
# SalmonesV3int8_bytetrack 72, 113
# SalmonesV13_botsort 74, 124
# SalmonesV13_bytetrack 75, 145
# SalmonesV13int8_botsort 33, 67
# SalmonesV13int8_bytetrack 36, 71
# SalmonesV33_botsort 114, 147
# SalmonesV33_bytetrack 107, 155
# SalmonesV33int8_botsort 52, 92
# SalmonesV33int8_bytetrack 46, 104

# SalmonesV4_botsort 99, 138
# SalmonesV4_bytetrack 99, 160
# SalmonesV4int8_botsort 55, 72
# SalmonesV4int8_bytetrack 61, 85
# SalmonesV14_botsort 90, 142
# SalmonesV14_bytetrack 93, 156
# SalmonesV14int8_botsort 55, 102
# SalmonesV14int8_bytetrack 58, 105
# SalmonesV44_botsort 94, 134
# SalmonesV44_bytetrack 86, 141
# SalmonesV44int8_botsort 55, 88
# SalmonesV44int8_bytetrack 52, 98

# Dataset:      Objetos detectados (V1 y V2)
# -------------------------------------------
# SalmonesV1:   323.0   252.0
# SalmonesV2:   94.5    163.5
# SalmonesV3:   106.5   176.5
# SalmonesV4:   99.0    149.0

# SalmonesV11:  192.5   183.0
# SalmonesV12:  83.0    157.0
# SalmonesV22:  81.5    142.0
# SalmonesV13:  74.5    134.5
# SalmonesV33:  110.5   151.0
# SalmonesV14:  91.5    149.0
# SalmonesV44:  90.0    137.5


# Casos con TensorRT
# Dataset:      Objetos detectados (V1 y V2)
# -------------------------------------------
# SalmonesV1:   188.0   185.5
# SalmonesV2:   51.5    94.0
# SalmonesV3:   69.0    106.0
# SalmonesV4:   58.0    78.5

# SalmonesV11:  141.5   123.5
# SalmonesV12:  57.0    98.0
# SalmonesV22:  64.0    109.0
# SalmonesV13:  34.5    69.0
# SalmonesV33:  49.0    98.0
# SalmonesV14:  56.5    103.5
# SalmonesV44:  53.5    93.0