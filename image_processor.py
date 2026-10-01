"""
Image Processor Module
Handles classical computer vision preprocessing:
- Resizing with aspect ratio preservation
- Grayscale conversion
- Gaussian blur for noise reduction
- Morphological Black-Hat transform to isolate localized dark road depressions (potholes)
- Adaptive statistical thresholding
- Morphological opening and closing for speckle noise suppression
- Canny edge detection for rim boundary extraction
- Intermediate pipeline stage generation
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any


class ImageProcessor:
    def __init__(self, target_width: int = 800):
        self.target_width = target_width

    def resize_with_aspect_ratio(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize image to standard target_width while preserving aspect ratio.
        Returns the resized image and the scaling factor.
        """
        h, w = image.shape[:2]
        if w <= self.target_width:
            return image.copy(), 1.0
        
        scale = self.target_width / float(w)
        target_height = int(h * scale)
        resized = cv2.resize(image, (self.target_width, target_height), interpolation=cv2.INTER_AREA)
        return resized, scale

    def process_pipeline(
        self,
        image: np.ndarray,
        gaussian_ksize: int = 7,
        canny_low: int = 40,
        canny_high: int = 120
    ) -> Dict[str, Any]:
        """
        Execute full classical computer vision pipeline and return all intermediate stages.
        """
        # Step 1: Resize preserving aspect ratio
        resized, scale = self.resize_with_aspect_ratio(image)
        h, w = resized.shape[:2]

        # Step 2: Grayscale conversion
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized.copy()

        # Step 3: Noise Reduction using Gaussian Blur
        ksize = gaussian_ksize if gaussian_ksize % 2 == 1 else gaussian_ksize + 1
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # Step 4: Morphological Black-Hat transform
        # Extracts dark depressions/craters that are darker than the local surrounding road
        structuring_radius = max(35, int(w * 0.05))
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (structuring_radius, structuring_radius))
        blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel_bg)

        # Step 5: Statistical thresholding on dark depression map
        # Computes threshold adaptively based on road surface texture variance
        mean_val = float(np.mean(blackhat))
        std_val = float(np.std(blackhat))
        thresh_val = max(24, int(mean_val + 1.85 * std_val))

        _, thresh_raw = cv2.threshold(blackhat, thresh_val, 255, cv2.THRESH_BINARY)

        # Step 6: Morphological opening and closing
        # Opening removes small road aggregate and gravel noise (5x5 kernel)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(thresh_raw, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # Closing seals potholes into solid coherent blobs (15x15 kernel)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # Clear borders (prevent frame borders from merging with contours)
        border_px = 14
        closed[0:border_px, :] = 0
        closed[-border_px:, :] = 0
        closed[:, 0:border_px] = 0
        closed[:, -border_px:] = 0

        # Step 7: Canny Edge Detection
        edges = cv2.Canny(blurred, canny_low, canny_high)

        return {
            "original_resized": resized,
            "scale": scale,
            "grayscale": gray,
            "blurred": blurred,
            "blackhat": blackhat,
            "threshold": closed,
            "edges": edges,
            "candidate_mask": closed,
            "dimensions": (w, h)
        }
