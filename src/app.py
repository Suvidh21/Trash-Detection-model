from pathlib import Path
import streamlit as st
from PIL import Image

import config
import app_helper

# Page setup
st.set_page_config(
    page_title="TerraScan — Smart Waste Detection",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Theme-compatible CSS)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
        background: linear-gradient(135deg, #10b981, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .tagline {
        font-size: 1.1rem;
        font-weight: 600;
        opacity: 0.70;
        margin-bottom: 0.3rem;
        letter-spacing: 0.5px;
    }
    .subtitle {
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
        opacity: 0.85;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section
st.markdown('<div class="main-title">🌍 TerraScan</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="tagline">Detect Waste. Decode Cities. Drive Change.</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Real-time object detection and categorization of waste items into <b>Recyclable</b>, <b>Non-Recyclable</b>, and <b>Hazardous</b> streams.</div>',
    unsafe_allow_html=True
)

# Sidebar Configuration
st.sidebar.title("⚙️ Detection Settings")

# Model Engine Selector
model_choice = st.sidebar.selectbox(
    "Select Vision Model Engine",
    config.MODEL_OPTIONS,
    index=0,
    help="Select 'YOLOv8 Standard' for high-accuracy bottle/cup/phone detection, or 'Custom Waste Model' for 21-class fine-grained waste taxonomy."
)

# Load Selected Model
try:
    model, model_type = app_helper.load_vision_model(model_choice)
except Exception as ex:
    st.error(f"Error loading vision model: {ex}")
    st.stop()

# Source Selection
source_radio = st.sidebar.radio(
    "Select Input Source",
    config.SOURCES_LIST,
    help="Choose whether to run detection on an uploaded image, video file, or live webcam."
)

# Confidence Slider
confidence = float(st.sidebar.slider(
    "Confidence Threshold (%)",
    min_value=15,
    max_value=90,
    value=35,
    help="Lower values detect more objects; higher values remove doubtful detections."
)) / 100.0

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Disposal Guidelines")
st.sidebar.markdown(
    """
    - **♻️ Recyclable**: Clean paper, cardboard, plastic bottles, metal cans.
    - **🗑️ Non-Recyclable**: Soiled items, food wrappers, plastic bags, straws.
    - **⚠️ Hazardous**: Batteries, electronics, spray cans, chemical containers.
    """
)

# --- MAIN CONTROLLER ---

if source_radio == config.IMAGE:
    st.subheader("📸 Image Detection")
    uploaded_file = st.file_uploader("Upload an image of waste items...", type=["jpg", "jpeg", "png", "webp"])

    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        try:
            input_image = Image.open(uploaded_file).convert("RGB")
            with col1:
                st.image(input_image, caption="Uploaded Image", width="stretch")

            with st.spinner("Analyzing image..."):
                output_image, detected_items = app_helper.process_image(model, input_image, confidence, model_type=model_type)

            with col2:
                st.image(output_image, caption="Detection Results", width="stretch")

            st.write("---")
            app_helper.render_detection_summary(detected_items, model_type=model_type)

        except Exception as e:
            st.error(f"Error processing image: {e}")
    else:
        st.info("Please upload an image to run waste detection.")

elif source_radio == config.VIDEO:
    st.subheader("📹 Video Detection")
    uploaded_video = st.file_uploader("Upload a video clip...", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        st.video(uploaded_video)
        if st.button("Start Processing Video"):
            with st.spinner("Processing video frames..."):
                app_helper.process_video(model, uploaded_video, confidence, model_type=model_type)
    else:
        st.info("Please upload a video file to run detection.")

elif source_radio == config.WEBCAM:
    st.subheader("🔴 Live Webcam Feed")
    app_helper.play_webcam(model, confidence, model_type=model_type)
