"""
Pothole Detector Module
Classical OpenCV Computer Vision Detector:
- Contour extraction and area/geometry analysis
- Aspect ratio, solidity, darkness contrast, and edge energy evaluation
- Filters out road markings, thin cracks, and noise
- Confidence scoring for each candidate
- Road condition classification (GOOD, MODERATE, POOR)
- High-visibility visual annotation with HUD overlay
- Video frame-by-frame processing and video generation
"""

import os
import time
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from image_processor import ImageProcessor


class PotholeDetector:
    def __init__(
        self,
        min_area: int = 900,
        max_area: int = 75000,
        min_aspect_ratio: float = 0.40,
        max_aspect_ratio: float = 3.60,
        min_solidity: float = 0.38,
        confidence_threshold: float = 50.0,
        moderate_pothole_threshold: int = 2,
        poor_pothole_threshold: int = 3
    ):
        """
        Configurable computer vision parameters for pothole detection.
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_solidity = min_solidity
        self.confidence_threshold = confidence_threshold
        
        # Road condition thresholds
        self.moderate_threshold = moderate_pothole_threshold
        self.poor_threshold = poor_pothole_threshold

        self.processor = ImageProcessor(target_width=800)

    def classify_road_condition(self, pothole_count: int) -> Dict[str, str]:
        """
        Classifies road condition based on detected pothole count.
        """
        if pothole_count == 0:
            return {
                "status": "GOOD",
                "color": "#10b981",  # emerald green
                "badge_class": "badge-good",
                "description": "Road surface is in good condition with no hazardous potholes detected."
            }
        elif pothole_count <= self.moderate_threshold:
            return {
                "status": "MODERATE",
                "color": "#f59e0b",  # amber yellow
                "badge_class": "badge-moderate",
                "description": f"Minor road distress detected ({pothole_count} pothole{'s' if pothole_count > 1 else ''}). Drivers should proceed with caution."
            }
        else:
            return {
                "status": "POOR",
                "color": "#ef4444",  # crimson red
                "badge_class": "badge-poor",
                "description": f"Hazardous road condition! {pothole_count} potholes detected. Immediate road maintenance required."
            }

    def calculate_candidate_features(
        self,
        contour: np.ndarray,
        gray_img: np.ndarray,
        edges_img: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Extract geometric, darkness, and edge features for a candidate contour.
        Returns None if candidate fails structural/pothole verification checks.
        """
        area = cv2.contourArea(contour)
        if area < self.min_area or area > self.max_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        if w < 20 or h < 20:
            return None

        aspect_ratio = float(w) / float(h)
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return None

        # Convex Hull and Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return None
        solidity = float(area) / float(hull_area)
        if solidity < self.min_solidity:
            return None

        # Compactness: 4*pi*area / perimeter^2 (rejects thin fragmented lines)
        peri = cv2.arcLength(contour, True)
        compactness = (4.0 * np.pi * area) / max(1.0, peri * peri)
        if compactness < 0.10:
            return None

        # Extent = area / bounding box area
        extent = float(area) / (w * h)
        if extent < 0.20 or extent > 0.95:
            return None

        # Intensity & Darkness contrast analysis
        mask = np.zeros(gray_img.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Dilate mask to sample surrounding asphalt
        kernel_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_mask = cv2.dilate(mask, kernel_ring, iterations=1)
        surround_ring = cv2.subtract(dilated_mask, mask)

        mean_inside = cv2.mean(gray_img, mask=mask)[0]
        mean_outside = cv2.mean(gray_img, mask=surround_ring)[0]

        darkness_contrast = max(0.0, (mean_outside - mean_inside) / max(1.0, mean_outside))

        # Edge energy: evaluate edge gradient around the perimeter of the pothole
        edge_density = cv2.mean(edges_img, mask=mask)[0] / 255.0

        # Pothole contrast verification:
        # A valid pothole either has notable depression contrast, or high edge boundary density (crater rim)
        if darkness_contrast < 0.04 and edge_density < 0.10:
            return None

        # Severity categorization based on area
        if area > 5000:
            severity = "HIGH"
        elif area > 2000:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # Calculate Confidence Score (0.0 to 100.0)
        # 1. Aspect ratio score: optimal range 0.7 to 2.2 -> max 25 points
        if 0.7 <= aspect_ratio <= 2.2:
            ar_score = 25.0
        else:
            ar_score = max(5.0, 25.0 - abs(aspect_ratio - 1.4) * 9.0)

        # 2. Solidity score: higher solidity = more coherent pothole depression -> max 25 points
        solidity_score = min(25.0, max(8.0, (solidity - 0.35) * 45.0))

        # 3. Darkness contrast score: darker depression than road surface -> max 30 points
        darkness_score = min(30.0, max(10.0, darkness_contrast * 70.0 + 10.0))

        # 4. Edge density score: well defined rim edges -> max 20 points
        edge_score = min(20.0, max(5.0, edge_density * 45.0 + 5.0))

        confidence = round(float(ar_score + solidity_score + darkness_score + edge_score), 1)
        confidence = min(98.5, max(45.0, confidence))

        if confidence < self.confidence_threshold:
            return None

        # Center point
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = x + w // 2
            cy = y + h // 2

        return {
            "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "center": {"x": cx, "y": cy},
            "area": int(area),
            "aspect_ratio": round(aspect_ratio, 2),
            "solidity": round(solidity, 2),
            "darkness_contrast": round(darkness_contrast, 2),
            "severity": severity,
            "confidence": confidence,
            "contour": contour
        }

    def detect_in_image(
        self,
        image_bgr: np.ndarray,
        save_pipeline_dir: Optional[str] = None,
        file_prefix: str = "img"
    ) -> Dict[str, Any]:
        """
        Performs full detection on an image array.
        Returns detection statistics, annotated image, and pipeline stages.
        """
        start_time = time.time()

        # Step 1: Preprocessing pipeline
        pipeline = self.processor.process_pipeline(image_bgr)
        resized = pipeline["original_resized"]
        gray = pipeline["grayscale"]
        blur = pipeline["blurred"]
        thresh = pipeline["threshold"]
        edges = pipeline["edges"]
        candidate_mask = pipeline["candidate_mask"]

        # Step 2: Contour detection
        contours, hierarchy = cv2.findContours(
            candidate_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Create intermediate visual for contours
        contours_vis = resized.copy()
        cv2.drawContours(contours_vis, contours, -1, (255, 120, 0), 2)

        # Step 3: Analyze each candidate contour
        raw_detections = []
        raw_contours = []

        # Sort contours by area descending
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in sorted_contours:
            features = self.calculate_candidate_features(cnt, gray, edges)
            if features is not None:
                raw_contours.append(cnt)
                raw_detections.append(features)

        # Step 3b: Spatial Proximity Merging
        # Unites fragmented crater rims, dark depressions, and cavity boundaries into single potholes
        valid_detections = []
        detection_contours = []
        used = set()

        for i in range(len(raw_detections)):
            if i in used:
                continue
            cur = dict(raw_detections[i])
            b = cur["bbox"]
            x1, y1, w, h = b["x"], b["y"], b["width"], b["height"]
            x2, y2 = x1 + w, y1 + h
            combined_cnts = [raw_contours[i]]
            merged_area = cur["area"]
            max_conf = cur["confidence"]

            for j in range(i + 1, len(raw_detections)):
                if j in used:
                    continue
                bj = raw_detections[j]["bbox"]
                jx1, jy1, jw, jh = bj["x"], bj["y"], bj["width"], bj["height"]
                jx2, jy2 = jx1 + jw, jy1 + jh

                dx = max(0, max(x1, jx1) - min(x2, jx2))
                dy = max(0, max(y1, jy1) - min(y2, jy2))

                # If bounding boxes are within 42 pixels of each other, merge into one pothole
                if dx <= 42 and dy <= 42:
                    x1 = min(x1, jx1)
                    y1 = min(y1, jy1)
                    x2 = max(x2, jx2)
                    y2 = max(y2, jy2)
                    merged_area += raw_detections[j]["area"]
                    max_conf = max(max_conf, raw_detections[j]["confidence"])
                    combined_cnts.append(raw_contours[j])
                    used.add(j)

            used.add(i)
            cur["bbox"] = {"x": int(x1), "y": int(y1), "width": int(x2 - x1), "height": int(y2 - y1)}
            cur["center"] = {"x": int(x1 + (x2 - x1) // 2), "y": int(y1 + (y2 - y1) // 2)}
            cur["area"] = int(merged_area)
            cur["confidence"] = round(float(max_conf), 1)

            # Severity categorization based on merged area
            if cur["area"] > 5000:
                cur["severity"] = "HIGH"
            elif cur["area"] > 2000:
                cur["severity"] = "MEDIUM"
            else:
                cur["severity"] = "LOW"

            feat_dict = {k: v for k, v in cur.items() if k != "contour"}
            valid_detections.append(feat_dict)
            detection_contours.append(np.vstack(combined_cnts))

        # Re-assign sequential detection IDs
        for idx, d in enumerate(valid_detections, 1):
            d["id"] = idx

        pothole_count = len(valid_detections)
        road_condition = self.classify_road_condition(pothole_count)
        
        avg_confidence = (
            round(sum(d["confidence"] for d in valid_detections) / pothole_count, 1)
            if pothole_count > 0 else 0.0
        )

        # Step 4: Draw High-Quality Visual Annotations
        annotated_img = self.draw_annotations(
            resized,
            valid_detections,
            detection_contours,
            road_condition,
            avg_confidence
        )

        elapsed_ms = round((time.time() - start_time) * 1000.0, 1)

        # Step 5: Save pipeline visuals if directory provided
        pipeline_urls = {}
        if save_pipeline_dir:
            os.makedirs(save_pipeline_dir, exist_ok=True)
            gray_path = os.path.join(save_pipeline_dir, f"{file_prefix}_gray.jpg")
            blur_path = os.path.join(save_pipeline_dir, f"{file_prefix}_blur.jpg")
            thresh_path = os.path.join(save_pipeline_dir, f"{file_prefix}_thresh.jpg")
            edges_path = os.path.join(save_pipeline_dir, f"{file_prefix}_edges.jpg")
            cont_path = os.path.join(save_pipeline_dir, f"{file_prefix}_contours.jpg")

            cv2.imwrite(gray_path, gray)
            cv2.imwrite(blur_path, blur)
            cv2.imwrite(thresh_path, thresh)
            cv2.imwrite(edges_path, edges)
            cv2.imwrite(cont_path, contours_vis)

            pipeline_urls = {
                "grayscale": f"/results/{os.path.basename(gray_path)}",
                "blur": f"/results/{os.path.basename(blur_path)}",
                "threshold": f"/results/{os.path.basename(thresh_path)}",
                "edges": f"/results/{os.path.basename(edges_path)}",
                "contours": f"/results/{os.path.basename(cont_path)}"
            }

        return {
            "success": True,
            "pothole_count": pothole_count,
            "road_condition": road_condition["status"],
            "road_condition_info": road_condition,
            "average_confidence": avg_confidence,
            "processing_time_ms": elapsed_ms,
            "detections": valid_detections,
            "annotated_image": annotated_img,
            "resized_image": resized,
            "pipeline_stages": pipeline_urls
        }

    def draw_annotations(
        self,
        base_img: np.ndarray,
        detections: List[Dict[str, Any]],
        contours: List[np.ndarray],
        road_condition: Dict[str, str],
        avg_confidence: float
    ) -> np.ndarray:
        """
        Draws high-visibility bounding boxes, translucent filled masks,
        and HUD status bar.
        """
        annotated = base_img.copy()
        h, w = annotated.shape[:2]

        # Draw semi-transparent filled masks on detected potholes
        overlay = annotated.copy()
        for cnt in contours:
            cv2.drawContours(overlay, [cnt], -1, (0, 0, 230), -1)  # crimson overlay
        cv2.addWeighted(overlay, 0.28, annotated, 0.72, 0, annotated)

        # Draw Bounding Boxes and Labels
        for det in detections:
            bbox = det["bbox"]
            x, y, bw, bh = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            conf = det["confidence"]
            det_id = det["id"]
            sev = det["severity"]

            if sev == "HIGH":
                color = (40, 40, 235)    # Red (BGR)
            elif sev == "MEDIUM":
                color = (20, 160, 245)   # Amber / Orange
            else:
                color = (0, 215, 255)    # Cyan / Yellow

            # Box border
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2, cv2.LINE_AA)
            
            # Corner accents
            corner_len = min(15, bw // 4, bh // 4)
            if corner_len > 4:
                # Top-left
                cv2.line(annotated, (x, y), (x + corner_len, y), (255, 255, 255), 3, cv2.LINE_AA)
                cv2.line(annotated, (x, y), (x, y + corner_len), (255, 255, 255), 3, cv2.LINE_AA)
                # Bottom-right
                cv2.line(annotated, (x + bw, y + bh), (x + bw - corner_len, y + bh), (255, 255, 255), 3, cv2.LINE_AA)
                cv2.line(annotated, (x + bw, y + bh), (x + bw, y + bh - corner_len), (255, 255, 255), 3, cv2.LINE_AA)

            # Center crosshair
            cx, cy = det["center"]["x"], det["center"]["y"]
            cv2.drawMarker(annotated, (cx, cy), color, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)

            # Pill Label Tag
            label_text = f"#{det_id} POTHOLE {conf}% [{sev}]"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.48
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            tag_y = max(y - 8, th + 8)
            tag_x = max(x, 2)

            # Tag background
            cv2.rectangle(
                annotated,
                (tag_x, tag_y - th - 5),
                (tag_x + tw + 8, tag_y + baseline),
                (15, 23, 42),
                -1
            )
            cv2.rectangle(
                annotated,
                (tag_x, tag_y - th - 5),
                (tag_x + tw + 8, tag_y + baseline),
                color,
                1
            )
            cv2.putText(
                annotated,
                label_text,
                (tag_x + 4, tag_y - 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        # HUD Top Status Banner
        banner_h = 36
        banner_overlay = annotated.copy()
        cv2.rectangle(banner_overlay, (0, 0), (w, banner_h), (11, 15, 25), -1)
        cv2.addWeighted(banner_overlay, 0.85, annotated, 0.15, 0, annotated)
        cv2.line(annotated, (0, banner_h), (w, banner_h), (50, 70, 95), 1)

        status_text = f"ROAD CONDITION: {road_condition['status']}  |  POTHOLES DETECTED: {len(detections)}"
        if len(detections) > 0:
            status_text += f"  |  AVG CONF: {avg_confidence}%"

        cond_bgr = (40, 200, 60) if road_condition["status"] == "GOOD" else (
            (20, 160, 245) if road_condition["status"] == "MODERATE" else (40, 40, 240)
        )

        cv2.circle(annotated, (16, 18), 6, cond_bgr, -1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            status_text,
            (30, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 245, 250),
            1,
            cv2.LINE_AA
        )

        return annotated

    def process_video(
        self,
        video_path: str,
        output_path: str,
        max_duration_sec: int = 30
    ) -> Dict[str, Any]:
        """
        Process uploaded video file frame-by-frame using OpenCV VideoCapture.
        Outputs an annotated MP4 video and summary statistics.
        """
        start_time = time.time()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_w = 800
        target_h = int(frame_height * (target_w / float(frame_width))) if frame_width > 0 else 600

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

        processed_count = 0
        max_potholes = 0
        potholes_history = []
        confidences_history = []
        first_annotated_frame = None

        max_frames_to_process = int(fps * max_duration_sec)

        while True:
            ret, frame = cap.read()
            if not ret or processed_count >= max_frames_to_process:
                break

            result = self.detect_in_image(frame)
            pothole_count = result["pothole_count"]
            potholes_history.append(pothole_count)
            if result["average_confidence"] > 0:
                confidences_history.append(result["average_confidence"])

            if pothole_count > max_potholes:
                max_potholes = pothole_count

            annotated_frame = cv2.resize(result["annotated_image"], (target_w, target_h))
            out_writer.write(annotated_frame)

            if first_annotated_frame is None and pothole_count > 0:
                first_annotated_frame = annotated_frame.copy()

            processed_count += 1

        cap.release()
        out_writer.release()

        if first_annotated_frame is None and processed_count > 0:
            first_annotated_frame = annotated_frame

        elapsed_sec = round(time.time() - start_time, 2)
        avg_potholes = round(sum(potholes_history) / max(1, len(potholes_history)), 2)
        avg_conf = round(sum(confidences_history) / max(1, len(confidences_history)), 1) if confidences_history else 0.0

        road_condition = self.classify_road_condition(max_potholes)

        return {
            "success": True,
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "fps": round(fps, 1),
            "duration_sec": round(processed_count / fps, 1) if fps > 0 else 0,
            "processing_time_sec": elapsed_sec,
            "max_potholes_detected": max_potholes,
            "average_potholes_per_frame": avg_potholes,
            "average_confidence": avg_conf,
            "road_condition": road_condition["status"],
            "road_condition_info": road_condition,
            "poster_frame": first_annotated_frame
        }
