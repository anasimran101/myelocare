import os
import shutil
import random
import yaml
import subprocess
import pandas as pd

# === CONFIG ===
base_dir = "/home/lab/code/diffusion_model/synthetic_orignal_merged"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")
output_dir = os.path.join(base_dir, "5fold")






# YOLOv8 settings
model_name = "yolov8n.pt"  
epochs = 300
img_size = 640
device = "0"  # GPU 0, or "cpu" if no GPU


# Define your classes
class_names = ["plasma_cell", "non_plasma_cell"]  
num_classes = len(class_names)

os.makedirs(output_dir, exist_ok=True)


















# === LOAD FILENAMES ===
images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
images.sort()
random.shuffle(images)

k = 5
fold_size = len(images) // k
results_summary = []

for i in range(k):
    fold_name = f"fold{i+1}"
    fold_path = os.path.join(output_dir, fold_name)

    train_img_dir = os.path.join(fold_path, "train/images")
    train_lbl_dir = os.path.join(fold_path, "train/labels")
    val_img_dir = os.path.join(fold_path, "val/images")
    val_lbl_dir = os.path.join(fold_path, "val/labels")

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # === Split data ===
    val_images = images[i * fold_size:(i + 1) * fold_size]
    train_images = [img for img in images if img not in val_images]

    def copy_files(file_list, dest_img, dest_lbl):
        for img in file_list:
            lbl = img.replace(".jpg", ".txt")
            img_src = os.path.join(images_dir, img)
            lbl_src = os.path.join(labels_dir, lbl)
            if os.path.exists(img_src) and os.path.exists(lbl_src):
                shutil.copy(img_src, os.path.join(dest_img, img))
                shutil.copy(lbl_src, os.path.join(dest_lbl, lbl))

    copy_files(train_images, train_img_dir, train_lbl_dir)
    copy_files(val_images, val_img_dir, val_lbl_dir)

    # === Write YAML ===
    yaml_path = os.path.join(output_dir, f"{fold_name}.yaml")
    yaml_data = {
        "train": os.path.join(fold_path, "train/images").replace("\\", "/"),
        "val": os.path.join(fold_path, "val/images").replace("\\", "/"),
        "nc": num_classes,
        "names": class_names,
    }

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f)

    print(f"\nCreated {fold_name}: {len(train_images)} train, {len(val_images)} val")
    print(f"YAML file: {yaml_path}")














# === Train YOLOv8 all folds ===
for i in range(k):
    fold_name = f"fold{i+1}"
    fold_path = os.path.join(output_dir, fold_name)

    train_img_dir = os.path.join(fold_path, "train/images")
    train_lbl_dir = os.path.join(fold_path, "train/labels")
    val_img_dir = os.path.join(fold_path, "val/images")
    val_lbl_dir = os.path.join(fold_path, "val/labels")
    yaml_path = os.path.join(output_dir, f"{fold_name}.yaml")
    print(f"Starting training for {fold_name} ...")
    run_name = f"fold{i+1}"
    run_dir = os.path.join(output_dir, "runs")
    cmd = [
        "yolo",
        "detect",
        "train",
        f"data={yaml_path}",
        f"model={model_name}",
        f"epochs={epochs}",
        f"imgsz={img_size}",
        f"device={device}",
        f"name={run_name}",
        f"project={run_dir}",
    
        # Stability & reproducibility
        f"seed={42+i}",
        f"exist_ok=True",
    
        # Performance
        f"batch=16",
        f"workers=4",
        f"amp=True",
    
        # Training quality
        f"patience=20",
        f"val=True",
    
        # Logging
        f"plots=True",
        f"save=True"
    ]
    subprocess.run(cmd, check=True)
        # === Extract metrics ===
    results_csv = os.path.join(run_dir, run_name, "results.csv")
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        last_row = df.iloc[-1]  
        results_summary.append({
            "Fold": fold_name,
            "Precision": last_row.get("metrics/precision(B)", None),
            "Recall": last_row.get("metrics/recall(B)", None),
            "mAP50": last_row.get("metrics/mAP50(B)", None),
            "mAP50-95": last_row.get("metrics/mAP50-95(B)", None)
        })
        print(f"Collected metrics for {fold_name}")
    else:
        print(f"Could not find results.csv for {fold_name}")




# === Save summary ===
summary_path = os.path.join(output_dir, "5fold_results.csv")
pd.DataFrame(results_summary).to_csv(summary_path, index=False)
print("\nCross-validation summary saved at:", summary_path)