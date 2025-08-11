from pathlib import Path
from ultralytics import YOLO
from datetime import date

# Repo root = one level up from /src
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "acne_detection.yaml"   # relative YAML
RUNS_DIR = ROOT / "runs"              # keep runs inside the repo

model = YOLO("yolov8s.pt")  # or 'yolov8s.yaml' to avoid downloading weights

results = model.train(
    data=str(DATA),
    imgsz=1280,
    epochs=130,
    batch=6,
    project=str(RUNS_DIR),                         # ensures outputs stay in repo
    name=f'y8s_acne_{date.today():%Y%m%d}',       # run name w/ date
    deterministic=True
)

 
# Training.
"""
- data: Path to the dataset YAML file.
- imgsz: The image size.pyth The default resolution is 640.
- epochs: Number of epochs we want to train for.
- patience: Number of epochs to wait without improvement in validation metrics before early stopping the training.
- batch: The batch size for data loader. You may increase or decrease it according to your GPU memory availability.
- name: Name of the results directory for runs/detect.
"""
