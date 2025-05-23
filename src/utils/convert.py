import json
import os
import argparse

from argparse import Namespace

# Globals
script_dir = os.path.dirname(__file__)
proj_root_dir = os.path.join(script_dir, '..', '..')
video_dir = os.path.join(proj_root_dir, 'data', 'videos')
label_data: list = []
CLASS_ID = 0

def convert_labelstudio_to_yolo(video_name: str):
    frames_dir = os.path.join(video_dir, video_name)

    frames_dir = frames_dir.rstrip(".mp4")
    os.makedirs(frames_dir, exist_ok=True)

    for task in label_data:
        annotations = task.get("annotations", [])
        if not annotations:
            continue

        result = annotations[0].get("result", [])

        frame_annotations = {}

        for ann in result:
            value = result[0].get("value")
            sequence = value.get("sequence", [])

            for seq in sequence:
                frame = seq["frame"]
                x_center = seq["x"] / 100
                y_center = seq["y"] / 100
                width = seq["width"] / 100
                height = seq["height"] / 100

                yolo_line = f"{CLASS_ID} {x_center} {y_center} {width} {height}"

                frame_filename = f"frame_{frame:05d}.txt"

                if frame_filename not in frame_annotations:
                    frame_annotations[frame_filename] = []
                frame_annotations[frame_filename].append(yolo_line)

        # Write to YOLO format
        for fname, lines in frame_annotations.items():
            with open(os.path.join(frames_dir, fname), "w") as f:
                f.write("\n".join(lines))


def extract_filenames(complete_filename: str) -> str:
    filename: str = ""
    if not complete_filename:
        return ""
    split_str = complete_filename.split("-")
    file_uuid = split_str[0]
    filename = complete_filename.lstrip(f"{file_uuid}-")
    return filename


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


def main(args: Namespace) -> None:
    labeled_videos: list = get_labeled_videos(args.label_file_path)
    for labeled_video in labeled_videos:
        convert_labelstudio_to_yolo(labeled_video)
    

def get_args() -> Namespace:
    parser = argparse.ArgumentParser(
        prog='Converter',
        description='Convert label-studio to yolo annotation.')
    parser.add_argument('-l', '--label_file_path', required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
