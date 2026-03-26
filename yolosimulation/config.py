from pathlib import Path
import torch

# ===== BASE PATH =====
BASE_PATH = Path.cwd()  # current working directory

# ===== DATASET PATHS =====
# DATASET_PATH = str(BASE_PATH / "datasets/MMDB/data/detection/flower")
# VAL_YAML_FILE = str(BASE_PATH / "datasets/MMDB/data/detection/patients/patient 01/patient01.yaml")
# MAIN_DATASET_PATH = str(BASE_PATH / "datasets/MMDB/data/detection")
# DATA_PATH = str(BASE_PATH / "datasets/MMDB/data/detection/train")

DATASET_PATH = str(BASE_PATH /          "datasets/merged_data/flower")
VAL_YAML_FILE = str(BASE_PATH /         "datasets/MMDB/data/detection/patients/patient 01/patient01.yaml")
MAIN_DATASET_PATH = str(BASE_PATH /     "datasets/merged_data")
DATA_PATH = str(BASE_PATH /             "datasets/merged_data")


# ===== OTHER CONFIG =====
NUM_CLIENTS = 3
BATCH_SIZE = 1
PARTITIONER = "none"
CLASSES = ["plasma", "non_plasma"]
LOCAL_EPOCHS = 1  # default local epochs for federated round

# ===== MODEL =====
MODEL_PATH = "pretrained_models/pmmdc_5epochs/best.pt"

# ===== DEVICE =====
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
STRATEGY_NAME = "FedAvg"
