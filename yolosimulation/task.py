# tasks.py
import os
import torch
from ultralytics import YOLO
from yolosimulation.config import *
import json
from datetime import datetime
from pathlib import Path
from flwr.common.typing import UserConfig


# ===== FUNCTIONS =====

def get_model():
    """
    Load YOLOv8 model and move to device.
    Returns a YOLO model object.
    """
    model = YOLO(MODEL_PATH)
    model.model.to(DEVICE)
    return model


def train(model, data_path, project = "runs", name = "train", lr=0.01, epochs=LOCAL_EPOCHS):
    """
    Train YOLOv8 model and return (metrics_dict, num_samples)
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"YOLOv8 YAML not found: {data_path}")

    # Count training samples from YAML or dataset
    from ultralytics.data.utils import check_det_dataset
    data_info = check_det_dataset(data_path)
    train_path = data_info.get("train", None)

    # Count images in train folder
    if train_path and os.path.isdir(train_path):
        os.listdir(train_path)
        num_samples = len([f for f in os.listdir(train_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    else:
        num_samples = 1  # fallback

    results = model.train(
        data=data_path,
        epochs=epochs,
        device=DEVICE,
        imgsz=640,
        batch=1,
        plots=True,
        project=project,
        name=name,
        verbose=False
    )

    metrics = results.results_dict
    metrics["num-examples"] = num_samples
    return metrics


def test(model,data_path, project = "runs", name = "train", device=DEVICE):
    """
    Evaluate YOLOv8 model on validation dataset.

    Returns:
        metrics: dict with main metrics
        num_samples: number of validation images
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"YOLOv8 {data_path} not found")

    results = model.val(data=data_path, device=device, project=project, name=name)
    metrics = results.results_dict

    # Count validation images
    val_path = None
    try:
        from ultralytics.data.utils import check_det_dataset
        data_info = check_det_dataset(data_path)
        val_path = data_info.get("val", None)
    except:
        pass

    if val_path and os.path.isdir(val_path):
        num_samples = len([f for f in os.listdir(val_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    else:
        num_samples = 1

    metrics["num-examples"] = num_samples
    return metrics


def make_project_name(train_or_val: str, client_or_server: str, round_num: int):
    """
    Returns folder for a specific round & type.
    """
    RUN_DIR = Path(os.environ["RUN_DIR"])
    project = RUN_DIR / train_or_val / f"round_{round_num}"
    project.mkdir(parents=True, exist_ok=True)
    return str(project), client_or_server


def create_run_dir(config: dict):
    """
    Create a folder for aggregated server results and save config.
    """
    RUN_DIR = Path(os.environ["RUN_DIR"])
    print(RUN_DIR)
    save_path = RUN_DIR / "server_aggregated_results"
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / "run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)

    return save_path, RUN_DIR