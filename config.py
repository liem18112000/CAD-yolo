import os

import ultralytics
from roboflow import Roboflow
from ultralytics import YOLO
from IPython import display


display.clear_output()
ultralytics.checks()

HOME = os.getcwd()
ROBOFLOW = Roboflow(api_key="xIfiwT8g9fPv9ZoJtmEh")
WORKSPACE = ROBOFLOW.workspace("cad-87e2z")
PROJECT = WORKSPACE.project("cad-lqngi")
VERSION = PROJECT.version(19)
