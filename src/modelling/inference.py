from ultralytics import YOLO # type: ignore

absolute_path = "/home/denis/Projects/var-ai/"

def detect_object_in_video(video_path, output_path, model_path=f"{absolute_path}models/pretrained/yolo11n.pt", save_output=True):
    model = YOLO(model_path)

    results = model(video_path, save=save_output, project=output_path)

    return results

def detect_human_poses_in_video(video_path, output_path, model_path=f"{absolute_path}models/pretrained/yolo11n-pose.pt", save_output=True):
    model = YOLO(model_path)

    results = model(video_path, save=save_output, project=output_path)

    return results