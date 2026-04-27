from ultralytics import YOLO
import torch

# 1. Load base pretrained model (architecture + initial weights)
model = YOLO("myelocare/pretrained_models/pmmdc_5epochs/best.pt")

# 2. Load your state_dict (fine-tuned / federated weights)
state_dict = torch.load(
    "myelocare/runs/20260327/run001/final_model.pt",
    map_location="cpu"
)

# If saved as {"model": state_dict}, unwrap it
if isinstance(state_dict, dict) and "model" in state_dict:
    state_dict = state_dict["model"]

# 3. Inject weights into YOLO internal model
missing, unexpected = model.model.load_state_dict(state_dict, strict=False)

print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

# 4. Fix metadata (IMPORTANT for correct predictions)
model.model.nc = 2  # from your earlier analysis
model.model.names = {0: "class0", 1: "class1"}

# 5. Run inference
image_path = "myelocare/datasets/MMDB/data/detection/patients/patient 01/images/IMG_20240608_102819.jpg"

results = model(
    image_path,
    device="cpu",
    save=True,
    project="myelocare/scripts",
    name="combined_model_inference"
)

print(results[0].boxes)
results[0].save(filename="myelocare/scripts/result.jpg")