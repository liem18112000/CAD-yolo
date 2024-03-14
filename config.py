import os

import ultralytics
from ultralytics import YOLO
from IPython import display


display.clear_output()
ultralytics.checks()

HOME = os.getcwd()
MODEL = YOLO(f'{HOME}/yolov8n.pt')

PREDICT_MODEL = YOLO(f"{HOME}/runs/detect/train/weights/best.pt")
