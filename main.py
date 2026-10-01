"""
FastAPI Main Application
Serves:
- REST API for classical OpenCV Pothole Detection on images and videos
- Static endpoints for generated results and uploaded files
- Frontend web dashboard
"""

import os
import uuid
import time
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from detector import PotholeDetector

# Define relative paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
SAMPLES_DIR = PROJECT_ROOT / "test_samples"

# Ensure runtime directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="AI-Based Road Pothole Detection Using Computer Vision",
    description="Classical OpenCV-based Pothole Detection REST API with FastAPI",
    version="1.0.0"
)

# CORS Middleware for modern web frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static asset folders
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/test_samples", StaticFiles(directory=str(SAMPLES_DIR)), name="test_samples")

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@app.get("/")
async def root():
    """
    Serve the frontend dashboard or welcome status.
    """
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "message": "AI-Based Road Pothole Detection Backend is Active",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health():
    """
    System health check endpoint.
    """
    return {
        "status": "healthy",
        "service": "AI-Road-Pothole-Detection",
        "version": "1.0.0",
        "backend": "FastAPI + OpenCV",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    min_area: int = Form(800),
    max_area: int = Form(75000),
    confidence_threshold: float = Form(50.0),
    moderate_threshold: int = Form(2),
    poor_threshold: int = Form(3)
):
    """
    Detect road potholes in an uploaded image using classical OpenCV techniques.
    """
    # Validate extension
    file_ext = Path(file.filename or "upload.jpg").suffix.lower()
    if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {file_ext}. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
        )

    # Read binary content
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Decode image via OpenCV
    np_arr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Please check file validity.")

    # Unique request ID
    uid = uuid.uuid4().hex[:10]
    orig_filename = f"orig_{uid}{file_ext}"
    annotated_filename = f"annotated_{uid}.jpg"

    orig_path = UPLOADS_DIR / orig_filename
    annotated_path = RESULTS_DIR / annotated_filename

    # Save original image
    with open(orig_path, "wb") as f:
        f.write(contents)

    # Initialize detector with user/default thresholds
    detector = PotholeDetector(
        min_area=min_area,
        max_area=max_area,
        confidence_threshold=confidence_threshold,
        moderate_pothole_threshold=moderate_threshold,
        poor_pothole_threshold=poor_threshold
    )

    # Run detection pipeline
    result = detector.detect_in_image(
        image_bgr=image_bgr,
        save_pipeline_dir=str(RESULTS_DIR),
        file_prefix=uid
    )

    # Save annotated result
    cv2.imwrite(str(annotated_path), result["annotated_image"])

    # Prepare response
    return {
        "success": True,
        "filename": file.filename,
        "processing_time_ms": result["processing_time_ms"],
        "pothole_count": result["pothole_count"],
        "road_condition": result["road_condition"],
        "road_condition_info": result["road_condition_info"],
        "average_confidence": result["average_confidence"],
        "annotated_image_url": f"/results/{annotated_filename}",
        "original_image_url": f"/uploads/{orig_filename}",
        "detections": result["detections"],
        "pipeline_stages": result["pipeline_stages"]
    }


@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    min_area: int = Form(500),
    confidence_threshold: float = Form(50.0)
):
    """
    Process an uploaded road video frame-by-frame using OpenCV.
    """
    file_ext = Path(file.filename or "upload.mp4").suffix.lower()
    if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {file_ext}. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        )

    uid = uuid.uuid4().hex[:10]
    orig_video_name = f"video_orig_{uid}{file_ext}"
    output_video_name = f"video_result_{uid}.mp4"
    poster_name = f"video_poster_{uid}.jpg"

    orig_video_path = UPLOADS_DIR / orig_video_name
    output_video_path = RESULTS_DIR / output_video_name
    poster_path = RESULTS_DIR / poster_name

    # Save uploaded video
    with open(orig_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detector = PotholeDetector(
        min_area=min_area,
        confidence_threshold=confidence_threshold
    )

    try:
        video_result = detector.process_video(
            video_path=str(orig_video_path),
            output_path=str(output_video_path),
            max_duration_sec=30
        )

        # Save poster frame if generated
        if video_result["poster_frame"] is not None:
            cv2.imwrite(str(poster_path), video_result["poster_frame"])
            poster_url = f"/results/{poster_name}"
        else:
            poster_url = None

        return {
            "success": True,
            "filename": file.filename,
            "total_frames": video_result["total_frames"],
            "processed_frames": video_result["processed_frames"],
            "fps": video_result["fps"],
            "duration_sec": video_result["duration_sec"],
            "processing_time_sec": video_result["processing_time_sec"],
            "max_potholes_detected": video_result["max_potholes_detected"],
            "average_potholes_per_frame": video_result["average_potholes_per_frame"],
            "average_confidence": video_result["average_confidence"],
            "road_condition": video_result["road_condition"],
            "road_condition_info": video_result["road_condition_info"],
            "video_result_url": f"/results/{output_video_name}",
            "poster_url": poster_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@app.get("/api/samples")
async def list_sample_images():
    """
    Returns list of preset sample images for demonstration and testing.
    """
    samples = []
    if SAMPLES_DIR.exists():
        for f in sorted(SAMPLES_DIR.iterdir()):
            if f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
                samples.append({
                    "name": f.stem.replace("_", " ").title(),
                    "filename": f.name,
                    "url": f"/test_samples/{f.name}"
                })
    return {"samples": samples}
