

<h1 align="center">🌍 TerraScan</h1>

<h3 align="center"><em>Detect Waste. Decode Cities. Drive Change.</em></h3>

<p align="center">
  <strong>AI-powered computer vision for smarter, cleaner cities</strong>
</p>

---

## 🎯 What is TerraScan?

**TerraScan** is an intelligent computer vision application built to tackle one of the most persistent challenges in metropolitan cities — **unmonitored waste accumulation on streets**.

In large urban centres, trash piles up on roads, sidewalks, and public spaces with no automated system to detect, report, or trigger cleanup. Municipal authorities rely on manual complaints that are slow, inconsistent, and rarely cover every neighbourhood equally.

**TerraScan changes that.**

By integrating with existing street-level and traffic surveillance cameras, TerraScan can:

- 🔍 **Detect** accumulated trash in real time using YOLOv8 object detection.
- 🗂️ **Identify & Segregate** waste into actionable categories — ♻️ Recyclable, 🗑️ Non-Recyclable, and ⚠️ Hazardous — promoting recycling and safe disposal at the source.
- 📍 **Report** detected waste with **actual images and geo-location tags** directly to local development authorities (LDAs), enabling rapid cleanup dispatch.
- ♻️ **Promote Recycling & Reuse** by surfacing recyclable materials that would otherwise end up in landfills, supporting circular-economy goals.

> **The vision:** Transform every traffic camera into a waste monitoring sentinel — making cities cleaner, faster, and more sustainable. 🏙️

---

## 🧠 Why TerraScan?

Urban areas face constant challenges with waste accumulation. Traditional reporting systems depend on citizens calling helplines or filing complaints — a process that is slow, uneven, and reactive.

TerraScan introduces a **smart, proactive trash detection system** that leverages computer vision and AI to:

- **Monitor** public areas using existing traffic and surveillance cameras.
- **Identify** trash build-up automatically — no human reporting required.
- **Alert** municipalities in real time so cleanup crews can respond faster.

This is a crucial step toward building **sustainable smart cities** — using infrastructure that already exists to solve problems that have persisted for decades. 🌍✨

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🖼️ **Image Detection** | Upload any photo to detect and classify waste objects |
| 📹 **Video Analysis** | Process video clips frame-by-frame for waste detection |
| 🔴 **Live Webcam Feed** | Real-time waste detection from connected cameras |
| 🧪 **Dual Model Support** | YOLOv8 Standard (COCO 80-class) + Custom 21-class waste taxonomy |
| ♻️ **Smart Segregation** | Auto-categorizes every detected item into Recyclable / Non-Recyclable / Hazardous |
| 📊 **Detection Dashboard** | Visual summary cards + detailed breakdown table with confidence scores |
| 🎛️ **Adjustable Confidence** | Fine-tune detection sensitivity via sidebar slider |

---

## 📁 Project Structure

```
TerraScan/
│
├── run_app.py                  # 🚀 Entry point — launch the app
├── requirements.txt            # 📦 Python dependencies
├── .gitignore                  # 🚫 Git exclusions
├── README.md                   # 📖 You are here
│
├── src/                        # 🧠 Application source code
│   ├── app.py                  #    Streamlit UI & page layout
│   ├── app_helper.py           #    Detection engine, video/webcam processing
│   └── config.py               #    Model paths, waste class mappings, settings
│
├── models/                     # 🤖 Model weights (not tracked in git)
│   └── weights/
│       ├── best.pt             #    Custom-trained waste detection model
│       └── yolov8n.pt          #    Pre-trained YOLOv8 Nano (COCO)
│
├── training/                   # 🏋️ Training scripts
│   ├── train_better_model.py   #    Multi-dataset download, merge & train pipeline
│   └── train_custom_model.py   #    Simple single-dataset training script
│
├── tests/                      # 🧪 Test & validation scripts
│   └── test_inference.py       #    Model loading + inference smoke test
│
├── notebooks/                  # 📓 Jupyter notebooks (experiments & EDA)
│   ├── YOLO_V3.ipynb
│   ├── train_colab.ipynb
│   ├── using_YOLOv5.ipynb
│   ├── yolo.ipynb
│   └── database_to_map_visualization.ipynb
│
├── data/                       # 📂 Datasets (not tracked in git)
│   └── GARBAGE_DATASET/
│
├── docs/                       # 📸 Documentation assets
│   ├── screenshots/            #    App screenshots
│   └── reference/              #    Reference images & diagrams
│
└── .streamlit/                 # 🎨 Streamlit theme configuration
    └── config.toml
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+** installed
- **pip** package manager
- A webcam (optional, for live detection)

### 1. Clone the Repository

```bash
git clone https://github.com/Suvidh21/Trash-Detection-model.git
cd Trash-Detection-model
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Weights

Place the trained model weights in the `models/weights/` directory:

| File | Description |
|------|-------------|
| `best.pt` | Custom-trained waste detection model (~112 MB) |
| `yolov8n.pt` | Pre-trained YOLOv8 Nano for COCO classes (~6.5 MB) |

> **Note:** Model weights are too large for GitHub. They will be shared separately or can be trained using the scripts in `training/`.

### 5. Launch the Application

```bash
python run_app.py
```

The app will open in your browser at **http://localhost:8501** 🎉

---

## 🎮 Usage

1. **Select a Model Engine** — Choose between YOLOv8 Standard (general objects) or Custom Waste Model (21-class fine-grained taxonomy) from the sidebar.
2. **Choose Input Source** — Upload an image, a video file, or start a live webcam feed.
3. **Adjust Confidence** — Use the slider to set the minimum detection confidence threshold.
4. **View Results** — Detected objects are highlighted on the image/video with bounding boxes, and a detailed segregation summary is displayed below.

---

## 🏋️ Training Your Own Model

### Quick Training (Local Dataset)

```bash
python training/train_custom_model.py
```

### Advanced Training (Multi-Dataset Pipeline)

1. Get a free API key from [Roboflow](https://app.roboflow.com).
2. Set your key in `training/train_better_model.py`.
3. Run:

```bash
python training/train_better_model.py
```

This will download 3 waste datasets, merge them, and train a YOLOv8s model with optimized hyperparameters. Trained weights are automatically copied to `models/weights/best.pt`.

---

## 🧪 Running Tests

```bash
python tests/test_inference.py
```

This runs a smoke test to verify model loading and inference.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **YOLOv8** (Ultralytics) | Real-time object detection |
| **Streamlit** | Interactive web UI |
| **OpenCV** | Image & video processing |
| **PyTorch** | Deep learning backend |
| **Roboflow** | Dataset management & augmentation |
| **Python 3.9+** | Core language |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Suvidh** — [GitHub](https://github.com/Suvidh21)

---

<p align="center">
  <strong>🌍 TerraScan — Detect Waste. Decode Cities. Drive Change.</strong><br/>
  <em>Making cities cleaner, one detection at a time.</em>
</p>
