# AI-Based Road Pothole Detection Using Computer Vision

A full-stack, local web application for automated road pothole inspection and road quality condition classification powered by **FastAPI** and **Classical OpenCV Computer Vision**.

---

## 📌 Project Overview

Road safety and timely infrastructure maintenance are critical for urban planning and driver safety. Potholes are structural failures in asphalt road surfaces caused by water penetration, thermal contraction/expansion, and heavy vehicle axle loads.

This project delivers an automated inspection system using **classical Computer Vision and Digital Image Processing (DIP)** techniques—without requiring heavy deep learning frameworks (such as YOLO, PyTorch, or TensorFlow) or external cloud APIs. It allows road safety inspectors and municipal engineers to upload road imagery or video feeds through an interactive dashboard to instantly detect, outline, count, and classify road conditions.

---

## ✨ Features

- **Classical Computer Vision Engine**: Pure Python, OpenCV, and NumPy processing pipeline.
- **Image & Video Support**: Detects potholes in static high-resolution photos and frame-by-frame video streams.
- **Bounding Boxes & High-Visibility Annotations**: Color-coded bounding boxes with corner accents, center crosshairs, translucent pothole cavity masks, and HUD banners.
- **Confidence Scoring**: Multi-metric candidate scoring based on geometry aspect ratio, contour solidity, darkness intensity contrast vs. surrounding road, and rim edge density.
- **Road Condition Classification**:
  - `0 Potholes` ➔ **GOOD** (Green)
  - `1 - 2 Potholes` ➔ **MODERATE** (Amber/Yellow)
  - `≥ 3 Potholes` ➔ **POOR** (Red / Hazardous)
- **Computer Vision Pipeline Visualizer**: Interactive step-by-step breakdown displaying intermediate OpenCV transformation stages (Grayscale, Gaussian Blur, Black-Hat Morphology, Canny Edges, Contours, and Final Result).
- **Interactive Dashboard UI**: Dark-mode glassmorphic interface, drag-and-drop zone, one-click preset test samples, parameter tuning sliders, side-by-side comparison, and direct result downloads.
- **RESTful API Architecture**: FastAPI backend with Swagger API documentation (`/docs`) and `/health` monitoring.

---

## 🛠️ Technology Stack

### Backend
- **Python 3.12+**
- **FastAPI**: Asynchronous high-performance REST API.
- **Uvicorn**: ASGI web server.
- **OpenCV (cv2)**: Image processing, morphological filtering, contour extraction, and video I/O.
- **NumPy**: Matrix transformations and statistical contrast calculations.
- **Python-Multipart & Aiofiles**: File streaming and asynchronous file handling.

### Frontend
- **HTML5 & CSS3**: Custom dark-mode glassmorphism theme.
- **JavaScript (Vanilla ES6+)**: Asynchronous fetch API, DOM manipulation, drag-and-drop.
- **Bootstrap 5**: Responsive grid and layout primitives.
- **Font Awesome 6**: Vector icons and visual indicators.

---

## 📂 Project Structure

```
AI-Road-Pothole-Detection/
│
├── backend/
│   ├── main.py              # FastAPI application & REST endpoints
│   ├── detector.py          # Classical OpenCV PotholeDetector class
│   ├── image_processor.py   # Preprocessing & CV pipeline stages
│   ├── requirements.txt     # Python dependencies
│   ├── uploads/             # Storage for uploaded images/videos
│   └── results/             # Generated annotated images, videos & stages
│
├── frontend/
│   ├── index.html           # Modern tech dashboard interface
│   ├── style.css            # Dark glassmorphic styling
│   └── script.js            # Frontend application logic & API controller
│
├── test_samples/            # Ready-to-use sample road images and video
│   ├── sample_1_moderate_road.jpg
│   ├── sample_2_poor_road.jpg
│   ├── sample_3_good_clean_road.jpg
│   └── sample_road_video.mp4
│
├── create_samples.py        # Synthetic road sample generator
├── test_detector.py         # Automated CLI detector verification script
├── README.md                # Comprehensive documentation & guide
└── .gitignore               # Git ignore rules
```

---

## 🔬 Computer Vision Pipeline Explained

The detector relies on classical optical and structural signatures of potholes on roadway asphalt:

```
Uploaded Image / Video Frame
            ↓
1. Standardized Resizing (Preserves aspect ratio to 800px width)
            ↓
2. Grayscale Conversion (3-channel BGR → 1-channel intensity)
            ↓
3. Gaussian Blur (7×7 kernel suppresses high-frequency asphalt aggregate noise)
            ↓
4. Morphological Black-Hat Transform (cv2.MORPH_BLACKHAT isolates sunken dark cavities)
            ↓
5. Statistical Thresholding (Adaptive threshold: Mean + 1.5 × StdDev)
            ↓
6. Morphological Opening & Closing (Cleans isolated road speckles & seals pothole blobs)
            ↓
7. Canny Edge Detection (Calculates rim gradient transitions)
            ↓
8. Contour Detection (cv2.findContours extracts boundary candidates)
            ↓
9. Candidate Feature Verification:
   ├── Minimum / Maximum Area check (Filters gravel & horizon)
   ├── Aspect Ratio check (Rejects long cracks & lane stripes)
   ├── Solidity check (Rejects branching/hollow noise)
   ├── Darkness Contrast (Interior depression vs. surrounding road ring)
   └── Edge Energy (Perimeter gradient density)
            ↓
10. Confidence Scoring & Road Condition Assessment
            ↓
11. HUD Overlay, Bounding Boxes & Rendered Result
```

---

## 🚀 Step-by-Step Execution Guide

### Prerequisites
- Windows 10/11 with Python 3.10+ installed and added to `PATH`.
- Modern web browser (Google Chrome, Edge, Firefox).

---

### Step 1: Open PowerShell in the Project Directory
Open **PowerShell** or VS Code Integrated Terminal in the project root:
```powershell
cd C:\Users\abish\OneDrive\Pictures\Documents\Desktop\AI-Road-Pothole-Detection
```

---

### Step 2: Activate the Virtual Environment
The virtual environment has already been initialized in `venv/`:
```powershell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run the command again).*

---

### Step 3: Install Dependencies (If not already installed)
```powershell
pip install -r backend/requirements.txt
```

---

### Step 4: Verify the Detector CLI Test
Run the automated detection script to verify the computer vision engine against pre-built test samples:
```powershell
python test_detector.py
```
Expected output:
- **Clean Good Road**: 0 potholes detected ➔ `GOOD`
- **Moderate Road**: 2 potholes detected ➔ `MODERATE`
- **Poor Road**: 5 potholes detected ➔ `POOR`

---

### Step 5: Start the FastAPI Backend Server
Run Uvicorn inside the `backend` directory:
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

---

### Step 6: Open the Web Application
Open your web browser and navigate to:
```
http://127.0.0.1:8000
```
or view the interactive Swagger API documentation at:
```
http://127.0.0.1:8000/docs
```

---

## 🧪 How to Test the Application

1. **One-Click Quick Presets**:
   - In the left upload card under **Quick Test Samples**, click:
     - `Moderate Road`: Automatically loads a 2-pothole test road.
     - `Poor Road`: Automatically loads a damaged road with multiple potholes.
     - `Clean Road`: Automatically loads a pristine asphalt road with 0 potholes.
     - `Test Video`: Automatically loads a road driving video sequence.
2. Click the **Detect Potholes** button.
3. Review the live results:
   - **Metrics Counter**: Pothole count, road condition, confidence %, and inference time in milliseconds.
   - **Visual Result**: High-resolution annotated image with bounding boxes and HUD.
   - **Side-by-Side Comparison**: Compare the original and annotated road frames.
   - **Detections Breakdown Table**: View bounding box coordinates, contour areas, and confidence ratings.
   - **CV Pipeline Stages**: Click any of the 7 stages (Grayscale, Blur, Black-Hat, Canny Edges, Contours) to inspect the OpenCV image transformations!
4. **Download Result**: Click "Download Result" to save the annotated inspection image or video.

---

## ⚠️ Limitations of Classical Computer Vision

While classical computer vision provides real-time processing without requiring heavy GPU hardware:
1. **Dynamic Lighting & Tree Shadows**: Sharp dark shadows cast by roadside trees or buildings can exhibit intensity profiles similar to asphalt depressions.
2. **Wet Asphalt & Puddles**: Water reflections alter road surface reflectance.
3. **Severe Surface Discoloration**: Fresh patches of dark asphalt on older, lighter asphalt may occasionally register higher darkness contrast.

---

## 🔮 Future Enhancements

- **Deep Learning Integration**: Incorporating lightweight YOLO (YOLOv8-nano / YOLOv11) or MobileNet-SSD to complement morphological feature extraction.
- **GPS Geotagging**: Embedding latitude and longitude coordinates in detection metadata for integration with GIS mapping systems.
- **Municipal Work-Order Integration**: Automatic ticket generation for road maintenance authorities.
- **Stereo Vision / Depth Sensors**: Estimating pothole depth and asphalt volume calculation using stereo cameras or LiDAR.
