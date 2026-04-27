from ultralytics import YOLO
import torch

# 1. Load base architecture (must match training model)
model = YOLO("myelocare/pretrained_models/pmmdc_5epochs/best.pt")

# 2. Load your state_dict

path = "myelocare/runs/20260327/run001"
state_dict = torch.load(
    path + "/final_model.pt",
    map_location="cpu"
)

# unwrap if needed
if isinstance(state_dict, dict) and "model" in state_dict:
    state_dict = state_dict["model"]

# 3. Load weights into YOLO model
model.model.load_state_dict(state_dict, strict=False)

# 4. IMPORTANT: fix metadata
model.model.nc = 2
model.model.names = {0: "class0", 1: "class1"}  # replace with real labels

# 5. SAVE as proper YOLO checkpoint
model.save(path + "/merged_final.pt" )

print("Saved merged YOLO model successfully.")