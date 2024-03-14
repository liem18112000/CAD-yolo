from config import MODEL
from roboflow import Roboflow

if __name__ == "__main__":
    rf = Roboflow(api_key="xIfiwT8g9fPv9ZoJtmEh")
    project = rf.workspace("cad-87e2z").project("cad-lqngi")
    version = project.version(2)
    dataset = version.download("yolov8")

    # Train
    MODEL.train(
        data=f"{dataset.location}/data.yaml",
        epochs=25,
        imgsz=640
    )



