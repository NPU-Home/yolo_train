from ultralytics import YOLO
import shutil
import os
from utils import *

MODEL = "yolov8"
SCALES = ['x', 'l', 'm']

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_TIME_IN_MIN = get_current_time_in_min()
VAL = True
SPLIT_DATASET = False
COPY_TO_ROOT = True

if SPLIT_DATASET:
    output_dataset_path, splits, classes = split_dataset(split_ratio="7:2:1", dataset_dir="dataset")
    make_dataset_yaml(dataset_path=output_dataset_path, splits=splits, classes=classes)

for scale in SCALES:
    model = YOLO(f"{MODEL}{scale}.pt")
    results = model.train(data="dataset.yaml", epochs=1000, batch=-1, device="0", name=f"{CURRENT_TIME_IN_MIN}_{scale}")
    
if VAL:
    for scale in SCALES:
        model = YOLO(f"runs/detect/{CURRENT_TIME_IN_MIN}_{scale}/weights/best.pt")
        results = model.val(data="dataset.yaml", batch=16, device="0", split='test', name=f"val_{CURRENT_TIME_IN_MIN}_{scale}")
        
if COPY_TO_ROOT:
    for scale in SCALES:
        pt_path = os.path.join(ROOT_DIR, f"runs/train/{CURRENT_TIME_IN_MIN}_{scale}/weights/best.pt")
        shutil.copy(pt_path, ROOT_DIR)