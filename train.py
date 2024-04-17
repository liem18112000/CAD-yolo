from ultralytics import YOLO

from config import HOME, PROJECT


if __name__ == "__main__":
    version = PROJECT.version(20)
    dataset = version.download("yolov9")
    location = dataset.location
    for model_name in [
        "yolov9c"
    ]:
        print(f"Model YOLO name: {model_name}")
        model = YOLO(f'{HOME}/models/yolov9/{model_name}.pt')
        model.train(
            data=f"{location}/data.yaml",
            epochs=300,
            imgsz=640,
            batch=10,
            patience=10
        )



