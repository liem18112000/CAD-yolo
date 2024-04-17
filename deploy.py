from config import HOME, PROJECT

version = PROJECT.version(20)
model_name = f"train18"
version.deploy("yolov9", f"{HOME}/runs/detect/{model_name}/")