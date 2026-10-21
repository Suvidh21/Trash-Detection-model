import os
import tempfile
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

import config


@st.cache_resource
def load_vision_model(model_choice):
    """
    Loads and caches the selected YOLO object detection model.
    """
    if "YOLOv8 Standard" in str(model_choice):
        coco_path = config.COCO_MODEL
        if coco_path.exists():
            return YOLO(str(coco_path)), "coco"
        return YOLO("yolov8n.pt"), "coco"
    else:
        path = config.CUSTOM_MODEL
        if path.exists():
            return YOLO(str(path)), "custom"
        return YOLO("yolov8n.pt"), "coco"


def format_class_name(name: str) -> str:
    """
    Converts raw snake_case model labels into clean title-cased names.
    e.g., 'plastic_bottle' -> 'Plastic Bottle'
    """
    return name.replace("_", " ").title()


def classify_waste_type(detected_items, model_type="custom"):
    """
    Categorizes detected items into Recyclable, Non-Recyclable, and Hazardous bins.
    Uses config.classify_label() so it works for any model's class names.
    """
    recyclable, non_recyclable, hazardous = [], [], []
    seen = set()
    for item in detected_items:
        if item in seen:
            continue
        seen.add(item)
        category = config.classify_label(item)
        if category == 'Recyclable':
            recyclable.append(item)
        elif category == 'Hazardous':
            hazardous.append(item)
        else:
            non_recyclable.append(item)
    return sorted(recyclable), sorted(non_recyclable), sorted(hazardous)


def render_detection_summary(detected_items_data, model_type="custom"):
    """
    Renders clean, theme-adaptive category cards and a detailed summary table.
    """
    # Filter out excluded classes (like 'person')
    valid_items = [item for item in detected_items_data if item["name"] not in config.COCO_EXCLUDE]

    if not valid_items:
        st.info("No waste items detected above the confidence threshold.")
        return

    item_names = [item["name"] for item in valid_items]
    recyclable, non_recyclable, hazardous = classify_waste_type(item_names, model_type)

    st.markdown("### 📊 Detection & Segregation Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div style="background-color: rgba(46, 125, 50, 0.08); border-left: 5px solid #2e7d32; padding: 14px; border-radius: 8px; margin-bottom: 12px;">
                <h4 style="color: #2e7d32; margin:0 0 8px 0; font-size: 16px;">♻️ Recyclable ({len(recyclable)})</h4>
                {"<ul style='margin:0; padding-left:20px; color:#1b5e20; font-weight:500;'>" + "".join([f"<li>{format_class_name(item)}</li>" for item in recyclable]) + "</ul>" if recyclable else "<p style='color:#888; margin:0;'>None detected</p>"}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="background-color: rgba(21, 101, 192, 0.08); border-left: 5px solid #1565c0; padding: 14px; border-radius: 8px; margin-bottom: 12px;">
                <h4 style="color: #1565c0; margin:0 0 8px 0; font-size: 16px;">🗑️ Non-Recyclable ({len(non_recyclable)})</h4>
                {"<ul style='margin:0; padding-left:20px; color:#0d47a1; font-weight:500;'>" + "".join([f"<li>{format_class_name(item)}</li>" for item in non_recyclable]) + "</ul>" if non_recyclable else "<p style='color:#888; margin:0;'>None detected</p>"}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div style="background-color: rgba(198, 40, 40, 0.08); border-left: 5px solid #c62828; padding: 14px; border-radius: 8px; margin-bottom: 12px;">
                <h4 style="color: #c62828; margin:0 0 8px 0; font-size: 16px;">⚠️ Hazardous ({len(hazardous)})</h4>
                {"<ul style='margin:0; padding-left:20px; color:#b71c1c; font-weight:500;'>" + "".join([f"<li>{format_class_name(item)}</li>" for item in hazardous]) + "</ul>" if hazardous else "<p style='color:#888; margin:0;'>None detected</p>"}
            </div>
            """,
            unsafe_allow_html=True
        )

    # Table breakdown
    st.markdown("#### Detected Objects Breakdown")
    table_data = []
    for idx, item in enumerate(valid_items, 1):
        raw_name = item["name"]
        clean_name = format_class_name(raw_name)

        category = config.classify_label(raw_name)
        if category == 'Recyclable':
            cat = "♻️ Recyclable"
        elif category == 'Hazardous':
            cat = "⚠️ Hazardous"
        else:
            cat = "🗑️ Non-Recyclable"

        table_data.append({
            "#": idx,
            "Detected Item": clean_name,
            "Category": cat,
            "Confidence": f"{item['confidence'] * 100:.1f}%"
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, width="stretch", hide_index=True)


def process_image(model, image: Image.Image, confidence: float, model_type="custom"):
    """
    Runs YOLO object detection on an image input.
    """
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model.predict(img_bgr, conf=confidence, imgsz=640)

    res_plotted = results[0].plot()
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    res_image = Image.fromarray(res_rgb)

    detected_items = []
    names = model.names
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        item_name = names[cls_id]
        detected_items.append({
            "name": item_name,
            "confidence": conf
        })

    return res_image, detected_items


def process_video(model, video_file, confidence: float, model_type="custom"):
    """
    Processes an uploaded video file frame-by-frame.
    """
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    stop_btn = st.button("Stop Video Processing")

    detected_classes_seen = set()

    while cap.isOpened():
        if stop_btn:
            break
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=confidence, imgsz=640)

        for result in results:
            names = model.names
            for c in result.boxes.cls:
                detected_classes_seen.add(names[int(c)])

        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        st_frame.image(res_rgb, caption="Live Video Processing", channels="RGB", width="stretch")

    cap.release()
    try:
        os.remove(tfile.name)
    except Exception:
        pass

    if detected_classes_seen:
        summary_items = [{"name": item, "confidence": 1.0} for item in detected_classes_seen]
        st.write("---")
        render_detection_summary(summary_items, model_type=model_type)


def play_webcam(model, confidence: float, model_type="custom"):
    """
    Runs live detection on local webcam feed.
    """
    st.info("Ensure your webcam is connected. Check the box below to start live feed.")

    run_cam = st.checkbox("Start Webcam Feed", value=False)

    if run_cam:
        cap = cv2.VideoCapture(config.WEBCAM_PATH)
        st_frame = st.empty()

        while run_cam and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break

            results = model.predict(frame, conf=confidence, imgsz=640)
            res_plotted = results[0].plot()

            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st_frame.image(res_rgb, channels="RGB", width="stretch")

        cap.release()
