import os
import sys
import cv2

# Add backend to path
sys.path.append(os.path.abspath("backend"))

from detector import PotholeDetector

detector = PotholeDetector(
    min_area=500,
    max_area=75000,
    confidence_threshold=50.0
)

samples = [
    ("Clean Good Road", "test_samples/sample_3_good_clean_road.jpg"),
    ("Moderate Road", "test_samples/sample_1_moderate_road.jpg"),
    ("Poor Road", "test_samples/sample_2_poor_road.jpg")
]

for label, path in samples:
    img = cv2.imread(path)
    if img is None:
        print(f"Error loading {path}")
        continue
    res = detector.detect_in_image(img, save_pipeline_dir="backend/results", file_prefix=label.lower().replace(" ", "_"))
    print(f"[{label}]")
    print(f"  Potholes Detected: {res['pothole_count']}")
    print(f"  Road Condition:    {res['road_condition']}")
    print(f"  Avg Confidence:    {res['average_confidence']}%")
    print(f"  Processing Time:   {res['processing_time_ms']} ms")
    for d in res['detections']:
        print(f"    - #{d['id']}: Area={d['area']}px, Box={d['bbox']}, Conf={d['confidence']}%, Sev={d['severity']}")
    print()
