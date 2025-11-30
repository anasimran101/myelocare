import os
import random
import shutil
import yaml
from yolosimulation.config import *



 
def create_partitions(dataset_dir, output_dir, no_of_clients, class_names, ratios):
    val_ratio = 0.2

    if len(ratios) != no_of_clients:
        raise ValueError(f"Ratios must match number of clients ({no_of_clients})")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    # If it exists, remove it
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create the directory again
    os.makedirs(output_dir)
    images_dir = os.path.join(dataset_dir, "train/images")
    labels_dir = os.path.join(dataset_dir, "train/labels")

    images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    images.sort()

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(images)

    total_images = len(images)
    start_idx = 0

    for i in range(no_of_clients):
        fold_name = f"client_{i}"
        fold_path = os.path.join(output_dir, fold_name)

        train_img_dir = os.path.join(fold_path, "train/images")
        train_lbl_dir = os.path.join(fold_path, "train/labels")
        val_img_dir = os.path.join(fold_path, "val/images")
        val_lbl_dir = os.path.join(fold_path, "val/labels")

        for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            os.makedirs(d, exist_ok=True)

        # Determine client slice
        client_count = int(ratios[i] * total_images)
        end_idx = start_idx + client_count
        client_images = images[start_idx:end_idx]

        # Split into train/val
        val_count = int(len(client_images) * val_ratio)
        val_images = client_images[:val_count]
        train_images = client_images[val_count:]

        # Copy files
        def copy_files(file_list, dest_img, dest_lbl):
            for img in file_list:
                lbl = img.replace(".jpg", ".txt")
                shutil.copy(os.path.join(images_dir, img), os.path.join(dest_img, img))
                shutil.copy(os.path.join(labels_dir, lbl), os.path.join(dest_lbl, lbl))

        copy_files(train_images, train_img_dir, train_lbl_dir)
        copy_files(val_images, val_img_dir, val_lbl_dir)

        # Write YAML
        yaml_path = os.path.join(output_dir, f"{fold_name}.yaml")
        yaml_data = {
            "train": train_img_dir.replace("\\", "/"),
            "val": val_img_dir.replace("\\", "/"),
            "nc": len(class_names),
            "names": class_names,
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)

        print(f"Created {fold_name}: {len(train_images)} train, {len(val_images)} val")

        # Move start index forward
        start_idx = end_idx

    # Derive train and val directories from VAL_YAML_FILE
    train_img_dir = str(Path(VAL_YAML_FILE).parent / "images").replace("\\", "/")

    # Create YAML data
    yaml_data = {
        "train": train_img_dir,
        "val": train_img_dir,
        "nc": len(class_names),
        "names": class_names,
    }

    # Write to YAML
    with open(VAL_YAML_FILE, "w") as f:
        yaml.dump(yaml_data, f)

