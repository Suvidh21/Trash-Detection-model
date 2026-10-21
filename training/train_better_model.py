"""
TerraScan — Smart Waste Dataset Downloader & Trainer
-----------------------------------------------------
Downloads 3 high-quality public waste datasets from Roboflow Universe,
merges them into a unified training set, and trains a YOLOv8 model.

Run this script locally (CPU - slow) OR copy to Google Colab for GPU training.

Requirements:
    pip install roboflow ultralytics opencv-python

How to use:
    1. Get your free Roboflow API key at https://app.roboflow.com (top-right > API keys)
    2. Set ROBOFLOW_API_KEY below
    3. Run: python training/train_better_model.py
"""

import os
import sys
import shutil
from pathlib import Path
from ultralytics import YOLO

# =============================================================
# CONFIGURATION
# =============================================================

# ✅ Paste your free Roboflow API key here
ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY"

# Training hyperparameters - tuned for waste detection
EPOCHS = 100          # 100 epochs minimum for 22-class model
IMG_SIZE = 640        # Standard YOLOv8 input size
BATCH_SIZE = 16       # Reduce to 8 if you run out of memory
BASE_MODEL = "yolov8s.pt"  # YOLOv8-Small: best accuracy/speed trade-off
RUN_NAME = "waste_v2_better"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "WASTE_DATASETS"

# =============================================================
# WASTE ITEM CLASS DEFINITIONS (22 fine-grained classes)
# Merged across all downloaded datasets
# =============================================================
WASTE_CLASSES = [
    "battery",
    "can",
    "cardboard_bowl",
    "cardboard_box",
    "chemical_plastic_bottle",
    "chemical_plastic_gallon",
    "chemical_spray_can",
    "light_bulb",
    "paint_bucket",
    "plastic_bag",
    "plastic_bottle",
    "plastic_bottle_cap",
    "plastic_box",
    "plastic_cultery",
    "plastic_cup",
    "plastic_cup_lid",
    "reuseable_paper",
    "scrap_paper",
    "scrap_plastic",
    "snack_bag",
    "stick",
    "straw",
]

# =============================================================
# DATASET DOWNLOAD (Roboflow Universe)
# =============================================================

def download_datasets():
    """
    Downloads multiple waste detection datasets from Roboflow Universe.
    Each dataset covers different waste item types and real-world conditions.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow...")
        os.system(f"{sys.executable} -m pip install roboflow")
        from roboflow import Roboflow

    if ROBOFLOW_API_KEY == "YOUR_ROBOFLOW_API_KEY":
        print("\n⚠️  ERROR: Please set your Roboflow API key in training/train_better_model.py")
        print("   Get a free key at: https://app.roboflow.com (Profile > API Keys)")
        sys.exit(1)

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    datasets_to_download = [
        {
            "workspace": "material-identification",
            "project": "garbage-classification-3",
            "version": 2,
            "desc": "21-class fine-grained waste classification (main)"
        },
        {
            "workspace": "divneet",
            "project": "taco-trash-annotations-in-context-of-everything",
            "version": 1,
            "desc": "TACO street trash dataset"
        },
        {
            "workspace": "yolov5-waste",
            "project": "waste-object-detection",
            "version": 2,
            "desc": "General waste object detection"
        },
    ]

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_paths = []

    for i, ds in enumerate(datasets_to_download):
        dest_folder = DATASET_DIR / f"ds_{i+1}"
        if dest_folder.exists():
            print(f"✅ Dataset {i+1} already downloaded at: {dest_folder}")
            downloaded_paths.append(dest_folder)
            continue

        print(f"\n📥 Downloading Dataset {i+1}: {ds['desc']}")
        print(f"   Workspace: {ds['workspace']} | Project: {ds['project']} | v{ds['version']}")

        try:
            project = rf.workspace(ds["workspace"]).project(ds["project"])
            version = project.version(ds["version"])
            dataset = version.download("yolov8", location=str(dest_folder), overwrite=True)
            print(f"   ✅ Downloaded to: {dest_folder}")
            downloaded_paths.append(dest_folder)
        except Exception as e:
            print(f"   ⚠️  Could not download dataset {i+1}: {e}")
            print("   Skipping this dataset and continuing with available ones...")

    return downloaded_paths


# =============================================================
# MERGE DATASETS
# =============================================================

def merge_datasets(dataset_paths):
    """
    Merges multiple YOLOv8-format datasets into a single unified dataset.
    All images and labels are copied into data/WASTE_DATASETS/merged/
    """
    merged_dir = DATASET_DIR / "merged"
    for split in ["train", "val", "test"]:
        (merged_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (merged_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    total_images = {"train": 0, "val": 0, "test": 0}

    for ds_idx, ds_path in enumerate(dataset_paths):
        ds_path = Path(ds_path)
        for split in ["train", "valid", "test"]:
            out_split = "val" if split == "valid" else split
            img_src = ds_path / split / "images"
            lbl_src = ds_path / split / "labels"

            if not img_src.exists():
                continue

            for img_file in img_src.glob("*"):
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                    continue

                new_stem = f"ds{ds_idx+1}_{img_file.stem}"
                new_name = new_stem + img_file.suffix
                dest_img = merged_dir / out_split / "images" / new_name

                shutil.copy2(img_file, dest_img)
                total_images[out_split] += 1

                lbl_file = lbl_src / (img_file.stem + ".txt")
                if lbl_file.exists():
                    dest_lbl = merged_dir / out_split / "labels" / (new_stem + ".txt")
                    shutil.copy2(lbl_file, dest_lbl)

    print(f"\n📦 Merged Dataset Summary:")
    print(f"   Train images: {total_images['train']}")
    print(f"   Val images:   {total_images['val']}")
    print(f"   Test images:  {total_images['test']}")

    return merged_dir


# =============================================================
# GENERATE YAML CONFIG
# =============================================================

def create_dataset_yaml(merged_dir: Path):
    """
    Creates a YAML config file for the merged dataset.
    """
    yaml_path = merged_dir / "data.yaml"
    classes_str = "\n".join([f"  - {c}" for c in WASTE_CLASSES])

    yaml_content = f"""# Merged Waste Detection Dataset
# Auto-generated by train_better_model.py

path: {merged_dir.as_posix()}
train: train/images
val: val/images
test: test/images

nc: {len(WASTE_CLASSES)}
names:
{classes_str}
"""
    yaml_path.write_text(yaml_content)
    print(f"\n📄 Dataset YAML written to: {yaml_path}")
    return yaml_path


# =============================================================
# TRAINING
# =============================================================

def train(yaml_path: Path):
    """
    Trains a YOLOv8s model with optimized hyperparameters for waste detection.
    """
    print("\n" + "="*60)
    print("🏋️  Starting YOLOv8 Training")
    print(f"   Model:   {BASE_MODEL}")
    print(f"   Epochs:  {EPOCHS}")
    print(f"   Batch:   {BATCH_SIZE}")
    print(f"   ImgSize: {IMG_SIZE}")
    print(f"   Dataset: {yaml_path}")
    print("="*60 + "\n")

    model = YOLO(BASE_MODEL)

    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=RUN_NAME,
        # Augmentation (helps generalize beyond training set conditions)
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        # Regularization
        dropout=0.1,
        weight_decay=0.0005,
        # Validation
        val=True,
        patience=30,
        save_period=20,
        # Logging
        plots=True,
        verbose=True,
    )

    best_weights = PROJECT_ROOT / "runs" / "detect" / RUN_NAME / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\n✅ Training complete!")
        print(f"   Best weights: {best_weights}")

        # Auto-copy the best weights into the models dir
        app_weights_dir = PROJECT_ROOT / "models" / "weights"
        app_weights_dir.mkdir(parents=True, exist_ok=True)
        dest = app_weights_dir / "best.pt"
        shutil.copy2(best_weights, dest)
        print(f"   ✅ Copied to app weights: {dest}")
        print(f"\n🚀 Restart the app to use the new model: python run_app.py")
    else:
        print("\n⚠️  Training finished but best.pt not found. Check runs/ folder.")


# =============================================================
# MAIN
# =============================================================

def main():
    print("="*60)
    print("🗂️  TerraScan — Smart Waste Model Training Pipeline")
    print("="*60)

    # Option A: Download fresh datasets from Roboflow Universe
    if "--use-existing" not in sys.argv:
        downloaded_paths = download_datasets()
    else:
        # Option B: Use the existing GARBAGE DATASET only
        downloaded_paths = [PROJECT_ROOT / "GARBAGE DATASET"]
        print(f"⚠️  Using existing dataset only: {downloaded_paths[0]}")

    if not downloaded_paths:
        print("❌ No datasets available. Please set your Roboflow API key and try again.")
        sys.exit(1)

    # Merge all downloaded datasets
    if len(downloaded_paths) > 1:
        merged_dir = merge_datasets(downloaded_paths)
    else:
        merged_dir = Path(downloaded_paths[0])

    # Create unified data.yaml
    yaml_path = create_dataset_yaml(merged_dir)

    # Train the model
    train(yaml_path)


if __name__ == "__main__":
    main()
