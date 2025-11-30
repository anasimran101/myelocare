#server

DATASET_PATH = "MMDB/data/detection/flower"
VAL_YAML_FILE = "MMDB/data/detection/patients/patient 01/patient01.yaml"
NUM_CLIENTS = 3
BATCH_SIZE = 1


#partiotioner

MAIN_DATASET_PATH = "MMDB/data/detection"
RUN_PARTITIONER = 1
CLASSES = ["plasma", "non_plasma"]
#task

# ===== CONFIG =====
MODEL_PATH = "pretrained_models/pmmdc_300epochs/best.pt"  # pretrained YOLOv8 model
DATA_PATH = "MMDB/data/detection/train"  # root dataset folder

import torch
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

LOCAL_EPOCHS = 1  # default local epochs for federated round
VAL_YAML_FILE = "MMDB/data/detection/patients/patient 01/patient01.yaml"




