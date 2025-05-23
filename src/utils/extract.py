import cv2
import os
import json
import argparse
from argparse import Namespace

import convert

# Globals
script_dir = os.path.dirname(__file__)
proj_root_dir = os.path.join(script_dir, '..', '..')
video_dir = os.path.join(proj_root_dir, 'data', 'videos')
label_data: list = []

def get_labeled_videos(rel_label_file_path: str) -> list:
    label_file_path: str = os.path.join(os.getcwd(), rel_label_file_path)
    try:
        with open(label_file_path, "r") as label_file:
            global label_data
            label_data = json.load(label_file)
    except Exception as e:
        print(f"Failed to load label file: {e}")
        return []

    labeled_videos: list = []

    for file in label_data:
        file_name: str = file.get("file_upload")
        file_name = extract_filenames(file_name)
        if file_name:
            labeled_videos.append(file_name)

    return labeled_videos


def extract_filenames(complete_filename: str) -> str:
    filename: str = ""
    if not complete_filename:
        return ""
    split_str = complete_filename.split("-")
    file_uuid = split_str[0]
    filename = complete_filename.lstrip(f"{file_uuid}-")
    return filename


def extract_frames(filename: str) -> None:
    video_path = os.path.join(video_dir, filename)

    filename = filename.rstrip(".mp4")
    frame_dir: str = os.path.join(video_dir, f"{filename}-frames")
    try:
        os.makedirs(frame_dir, exist_ok=True)
    except Exception as e:
        print(f"Frames directory for {filename} already exists: {e}")

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frame_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1
    cap.release()


def main(args: Namespace) -> None:
    labeled_videos: list = get_labeled_videos(args.label_file_path)
    for labeled_video in labeled_videos:
        extract_frames(labeled_video)


def get_args() -> Namespace:
    parser = argparse.ArgumentParser(
        prog='Extractor',
        description='Extract frames from annotated videos.')
    parser.add_argument('-l', '--label_file_path', required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
