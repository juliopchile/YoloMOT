# This code is just to check if data saved in the MOTchallenge format correctly.
# We use the labels saved in MOTchallenge format to then visualice said labels in the corresponding images.

from utils import draw_detections, draw_segmentations

video_path = "dataset/source/people_walking_1_200"

detections_path = "data/trackers/mot_challenge/MyCustomChallenge-test/yolo11x_botsort/data/people_walking_1_200.txt"
detections_output_path="visualization_example/people_walking_1_200"

segmentations_path = "output/people_walking_1_200_seg.txt"
segmentations_output_path="output_visualization_seg/people_walking_1_200"

draw_detections(video_path, detections_path, detections_output_path)
# draw_segmentations(video_path, segmentations_path, segmentations_output_path)
# this function works but I don't have an example data to use right now so I just commented it out