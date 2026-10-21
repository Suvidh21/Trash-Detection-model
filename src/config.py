from pathlib import Path
import sys

# Resolve the project root (parent of the `src/` directory)
file_path = Path(__file__).resolve()
SRC_DIR = file_path.parent
ROOT = SRC_DIR.parent  # Project root

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# ML Model paths — now under models/weights/
MODEL_DIR = ROOT / 'models' / 'weights'
CUSTOM_MODEL = MODEL_DIR / 'best.pt'
COCO_MODEL = MODEL_DIR / 'yolov8n.pt'

MODEL_COCO = "YOLOv8 Standard (Highly Accurate for Bottles, Cups, Phones)"
MODEL_CUSTOM = "Custom Waste Model (Trained: Paper, Plastic, Glass, Metal, Cardboard, Biodegradable)"

MODEL_OPTIONS = [MODEL_COCO, MODEL_CUSTOM]

# Input Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM]

# Default detection settings
DEFAULT_CONFIDENCE = 0.35
WEBCAM_PATH = 0

# --- Custom Model Waste Mappings (21-class fine-grained model) ---
RECYCLABLE = ['cardboard_box', 'can', 'plastic_bottle_cap', 'plastic_bottle', 'reuseable_paper']
NON_RECYCLABLE = ['plastic_bag', 'scrap_paper', 'stick', 'plastic_cup', 'snack_bag', 'plastic_box', 'straw', 'plastic_cup_lid', 'scrap_plastic', 'cardboard_bowl', 'plastic_cultery']
HAZARDOUS = ['battery', 'chemical_spray_can', 'chemical_plastic_bottle', 'chemical_plastic_gallon', 'light_bulb', 'paint_bucket']

# --- Broad-category model mappings (garbage-classification-3 retrained model) ---
# Classes: PAPER, PLASTIC, GLASS, METAL, CARDBOARD, CLOTH, BIODEGRADABLE
BROAD_RECYCLABLE = ['PAPER', 'PLASTIC', 'GLASS', 'METAL', 'CARDBOARD',
                    'paper', 'plastic', 'glass', 'metal', 'cardboard']
BROAD_NON_RECYCLABLE = ['CLOTH', 'BIODEGRADABLE',
                         'cloth', 'biodegradable']
BROAD_HAZARDOUS = []  # none in broad category dataset

def classify_label(label: str) -> str:
    """
    Returns 'Recyclable', 'Non-Recyclable', or 'Hazardous' for any
    detected class label, regardless of which model produced it.
    """
    lbl = label.lower().strip()
    # Broad category checks
    for cls in BROAD_RECYCLABLE:
        if lbl == cls.lower():
            return 'Recyclable'
    for cls in BROAD_NON_RECYCLABLE:
        if lbl == cls.lower():
            return 'Non-Recyclable'
    # Fine-grained checks
    if label in RECYCLABLE:
        return 'Recyclable'
    if label in NON_RECYCLABLE:
        return 'Non-Recyclable'
    if label in HAZARDOUS:
        return 'Hazardous'
    # COCO fallback
    if label in COCO_RECYCLABLE:
        return 'Recyclable'
    if label in COCO_NON_RECYCLABLE:
        return 'Non-Recyclable'
    if label in COCO_HAZARDOUS:
        return 'Hazardous'
    return 'Non-Recyclable'  # safe default

# --- COCO Standard Waste Mappings ---
COCO_RECYCLABLE = ['bottle', 'wine glass', 'book']
COCO_NON_RECYCLABLE = ['cup', 'fork', 'knife', 'spoon', 'bowl', 'apple', 'banana', 'orange', 'donut', 'cake', 'sandwich']
COCO_HAZARDOUS = ['cell phone', 'laptop', 'mouse', 'keyboard', 'remote', 'tv', 'microwave', 'toaster', 'clock', 'scissors']
COCO_EXCLUDE = ['person']
