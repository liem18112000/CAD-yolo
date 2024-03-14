from config import PREDICT_MODEL
from roboflow import Roboflow
from IPython.display import display, Image

if __name__ == "__main__":
    rf = Roboflow(api_key="xIfiwT8g9fPv9ZoJtmEh")
    project = rf.workspace("cad-87e2z").project("cad-lqngi")
    version = project.version(2)
    dataset = version.download("yolov8")

    image = "part12_j-2_-_part13_oO_png.rf.01ac17dc67de6cc38c112d1c1aeb39e0.jpg"

    # predict
    PREDICT_MODEL.predict(
        source=f"{dataset.location}/train/images/{image}",
        conf=0.25
    )

    Image(filename=f'{dataset.location}/detect/predict/{image}', height=600)



