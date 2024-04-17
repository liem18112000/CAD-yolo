from ultralytics import YOLO

from config import DATASET, HOME
from roboflow import Roboflow
from IPython.display import display, Image

PREDICT_MODEL = YOLO(f"{HOME}/runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    image = "part12_j-2_-_part13_oO_png.rf.01ac17dc67de6cc38c112d1c1aeb39e0.jpg"

    # predict
    PREDICT_MODEL.predict(
        source=f"{DATASET.location}/train/images/{image}",
        conf=0.25
    )

    Image(filename=f'{DATASET.location}/detect/predict/{image}', height=600)



