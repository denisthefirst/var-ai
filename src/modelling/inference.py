import os
from ultralytics import YOLO  # type: ignore

root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))

"""
Module for running inferences on various models used in this project
"""

def detect_object_in_video(video_path, output_path, save_output=True):
    """
    Detects objects in a video using a YOLO object detection model.

    Args:
        video_path (str): The path to the video file to be processed.
        output_path (str): The directory path where the output (e.g., detection results) will be saved.
        save_output (bool, optional): Flag to save the output results. Defaults to True.

    Returns:
        results: The results from the YOLO model, typically including detected objects, their locations, and confidence scores.
    """
    global root_path
    model_path = f"{root_path}/models/pretrained/yolo11n.pt"
    model = YOLO(model_path)
    results = model(video_path, save=save_output, project=output_path)
    return results


def detect_human_poses_in_video(video_path, output_path, save_output=True):
    """
    Detects human poses in a video using a YOLO model designed for pose estimation.

    Args:
        video_path (str): The path to the video file to be processed.
        output_path (str): The directory path where the output (e.g., pose estimation results) will be saved.
        save_output (bool, optional): Flag to save the output results. Defaults to True.

    Returns:
        results: The results from the YOLO model, typically including the detected poses and keypoints.
    """
    global root_path
    model_path = f"{root_path}/models/pretrained/yolo11n-pose.pt"
    model = YOLO(model_path)
    results = model(video_path, save=save_output, project=output_path)
    return results
