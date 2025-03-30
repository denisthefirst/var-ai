from ultralytics import YOLO


def detect_object_in_video(video_path, output_path, model_path="../../models/pretrained/yolov8n.pt", save_output=True):
    model = YOLO(model_path)

    results = model(video_path, save=save_output, project=output_path)

    return results