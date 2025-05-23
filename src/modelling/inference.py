import os
from ultralytics import YOLO  # type: ignore

script_dir = os.path.dirname(__file__)
proj_root_dir = os.path.join(script_dir, '..', '..')
video_download_path = os.path.join(proj_root_dir, 'data', 'raw')
object_detected_video = os.path.join(proj_root_dir, 'data', 'object_detected')

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
    model_path = f"{proj_root_dir}/models/pretrained/yolo11s.pt"
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
    model_path = f"{proj_root_dir}/models/pretrained/yolo11n-pose.pt"
    model = YOLO(model_path)
    results = model(video_path, save=save_output, project=output_path)
    return results

def scan_videos_dir() -> list:
    """
    This function returns all filenames inside the download
    directory.

    :returns: A list with filenames and file extension.
    """
    try:
        return os.listdir(video_download_path)
    except Exception as e:
        print(f"Error reading download directory: {e}")
        return []


def run_object_detection_on_all_videos():
    """
    Run object detection on all videos in the download directory that haven't been processed yet.
    
    Returns:
        None
    """
    downloaded_files = scan_videos_dir() 
    
    for file in downloaded_files:
        video_path = os.path.join(video_download_path, file)
        print(f"Processing: {file}")
        detect_object_in_video(video_path, object_detected_video)

def main():
    run_object_detection_on_all_videos()


if __name__ == "__main__":
    main()