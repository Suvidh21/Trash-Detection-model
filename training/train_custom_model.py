from pathlib import Path
from ultralytics import YOLO

def main():
    print("==================================================")
    print("🏋️  TerraScan — YOLOv8 Model Training Script")
    print("==================================================")

    project_root = Path(__file__).resolve().parent.parent
    dataset_yaml = project_root / "GARBAGE DATASET" / "data.yaml"
    if not dataset_yaml.exists():
        print(f"Error: Dataset data.yaml not found at {dataset_yaml}")
        return

    print(f"Dataset path: {dataset_yaml}")
    print("Initializing YOLOv8 Small model...")

    # Load base YOLOv8 model
    model = YOLO("yolov8s.pt")

    # Train for 50 epochs
    print("Starting model training (50 epochs)...")
    results = model.train(
        data=str(dataset_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        name="waste_detection_model"
    )

    print("✅ Training complete! Best weights saved to 'runs/detect/waste_detection_model/weights/best.pt'")

if __name__ == "__main__":
    main()
