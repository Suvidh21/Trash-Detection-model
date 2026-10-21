import sys
from pathlib import Path
import cv2
import numpy as np

# Resolve project root and add src/ to sys.path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
sys.path.append(str(src_dir))

import config
import app_helper

def test_model_loading_and_inference():
    print("--- 1. Testing Model Weights Loading ---")
    custom_path = config.CUSTOM_MODEL
    print(f"Custom model path: {custom_path}")
    assert custom_path.exists(), f"Model file not found at {custom_path}"

    model, mtype = app_helper.load_vision_model(config.MODEL_COCO)
    print(f"✅ COCO Model loaded successfully! Type: {mtype}")

    print("\n--- 2. Testing Sample Inference ---")
    test_img_dir = root_dir / "data" / "GARBAGE_DATASET" / "test" / "images"
    if not test_img_dir.exists():
        # Fallback to old location
        test_img_dir = root_dir / "GARBAGE DATASET" / "test" / "images"

    sample_images = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))

    if sample_images:
        sample_path = sample_images[0]
        print(f"Using test sample image: {sample_path}")
        img = cv2.imread(str(sample_path))
    else:
        print("Creating synthetic test image...")
        img = np.zeros((480, 640, 3), dtype=np.uint8)

    results = model.predict(img, conf=0.2)
    names = model.names
    print(f"✅ Prediction completed. Detections found: {len(results[0].boxes)}")

    detected_items = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        item_name = names[cls_id]
        detected_items.append(item_name)
        print(f"  - Detected: '{item_name}' (conf: {conf:.2f})")

    recyclable, non_recyclable, hazardous = app_helper.classify_waste_type(detected_items, model_type="coco")
    print(f"\n--- 3. Segregation Results ---")
    print(f"  ♻️ Recyclable ({len(recyclable)}): {recyclable}")
    print(f"  🗑️ Non-Recyclable ({len(non_recyclable)}): {non_recyclable}")
    print(f"  ⚠️ Hazardous ({len(hazardous)}): {hazardous}")
    print("\n✅ All tests passed successfully!")

if __name__ == "__main__":
    test_model_loading_and_inference()
