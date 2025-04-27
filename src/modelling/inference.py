import os
from ultralytics import YOLO  # type: ignore

root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))


def detect_object_in_video(video_path, output_path, save_output=True):
    global root_path
    model_path = f"{root_path}/models/pretrained/yolo11n.pt"
    model = YOLO(model_path)
    results = model(video_path, save=save_output, project=output_path)
    return results


def detect_human_poses_in_video(video_path, output_path, save_output=True):
    global root_path
    model_path = f"{root_path}/models/pretrained/yolo11n-pose.pt"
    model = YOLO(model_path)
    results = model(video_path, save=save_output, project=output_path)
    return results
