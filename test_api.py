import os
import requests
import json

# Test API endpoints
base_url = "http://127.0.0.1:8000"

# 1. Health
r_health = requests.get(f"{base_url}/health")
print("Health:", r_health.status_code, r_health.json())

# 2. Detect Image
sample_path = "test_samples/sample_1_moderate_road.jpg"
with open(sample_path, "rb") as f:
    files = {"file": ("sample_1_moderate_road.jpg", f, "image/jpeg")}
    r_img = requests.post(f"{base_url}/detect/image", files=files)

print("Detect Image:", r_img.status_code)
res = r_img.json()
print("Success:", res.get("success"))
print("Potholes detected:", res.get("pothole_count"))
print("Road condition:", res.get("road_condition"))
print("Annotated image URL:", res.get("annotated_image_url"))
print("Detections count:", len(res.get("detections", [])))
print("Pipeline stages:", list(res.get("pipeline_stages", {}).keys()))

# 3. Detect Video
video_path = "test_samples/sample_road_video.mp4"
with open(video_path, "rb") as f:
    files = {"file": ("sample_road_video.mp4", f, "video/mp4")}
    r_vid = requests.post(f"{base_url}/detect/video", files=files)

print("Detect Video:", r_vid.status_code)
v_res = r_vid.json()
print("Video Success:", v_res.get("success"))
print("Frames:", v_res.get("processed_frames"))
print("Max potholes in frame:", v_res.get("max_potholes_detected"))
print("Video Road condition:", v_res.get("road_condition"))
print("Video Result URL:", v_res.get("video_result_url"))
