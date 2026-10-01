"""
Synthetic Realistic Road & Pothole Image Generator
Creates realistic asphalt texture, lane markings, and irregular pothole cavities
for immediate testing and demo presets.
"""

import os
import cv2
import numpy as np


def generate_asphalt_texture(width=800, height=600):
    """Generates realistic grainy asphalt roadway."""
    # Base road gray
    base_color = 115
    road = np.full((height, width, 3), base_color, dtype=np.uint8)

    # Add multiscale Perlin-like asphalt grain noise
    np.random.seed(42)
    noise1 = np.random.normal(0, 14, (height, width, 3)).astype(np.int16)
    noise2 = cv2.resize(np.random.normal(0, 18, (height // 4, width // 4, 3)).astype(np.int16), (width, height))
    
    road_grain = np.clip(road.astype(np.int16) + noise1 + noise2, 40, 210).astype(np.uint8)

    # Road perspective shading (slightly darker near bottom, lighter towards horizon)
    gradient = np.linspace(0.85, 1.15, height)[:, np.newaxis, np.newaxis]
    road_shaded = np.clip(road_grain * gradient, 0, 255).astype(np.uint8)

    # Road lane markings (yellow center line, white edge lines)
    # Edge lines
    cv2.line(road_shaded, (60, 0), (20, height), (220, 220, 220), 8, cv2.LINE_AA)
    cv2.line(road_shaded, (width - 60, 0), (width - 20, height), (220, 220, 220), 8, cv2.LINE_AA)

    # Dashed center line
    dash_len = 50
    gap_len = 40
    y = 20
    center_x = width // 2
    while y < height:
        cv2.line(road_shaded, (center_x, y), (center_x, min(height, y + dash_len)), (40, 210, 240), 6, cv2.LINE_AA)
        y += dash_len + gap_len

    return road_shaded


def draw_realistic_pothole(img, center_x, center_y, radius_x, radius_y, angle=0):
    """Renders a realistic dark sunken road depression with rough irregular perimeter."""
    # Generate irregular polygon for pothole rim
    num_pts = 32
    angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
    
    # Perturb radii with noise to make natural rough rim
    np.random.seed(int(center_x * 10 + center_y))
    radii_variation = np.random.uniform(0.75, 1.25, num_pts)
    
    pts = []
    for a, r_var in zip(angles, radii_variation):
        rx = radius_x * r_var * np.cos(a)
        ry = radius_y * r_var * np.sin(a)
        
        # Rotate by angle
        rad_ang = np.radians(angle)
        x_rot = rx * np.cos(rad_ang) - ry * np.sin(rad_ang)
        y_rot = rx * np.sin(rad_ang) + ry * np.cos(rad_ang)
        
        pts.append([int(center_x + x_rot), int(center_y + y_rot)])

    pts = np.array(pts, dtype=np.int32)

    # Shadow depression mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # Outer rough rim gradient (slightly lighter fractured asphalt edge)
    rim_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    rim_ring = cv2.subtract(rim_mask, mask)
    
    # Apply rim darkening & texture
    img[rim_ring > 0] = np.clip(img[rim_ring > 0].astype(np.int16) - 30, 0, 255).astype(np.uint8)

    # Inner deep crater (significantly darker)
    pothole_interior = np.full_like(img, 38)
    interior_noise = np.random.normal(0, 12, img.shape).astype(np.int16)
    pothole_texture = np.clip(pothole_interior.astype(np.int16) + interior_noise, 15, 75).astype(np.uint8)

    # Smooth blur the mask for soft depth transition
    blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0)
    alpha = (blurred_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

    img[:] = ((1.0 - alpha) * img.astype(np.float32) + alpha * pothole_texture.astype(np.float32)).astype(np.uint8)

    # Add small scattered pebble highlights inside pothole
    for _ in range(8):
        px = int(center_x + np.random.uniform(-radius_x * 0.5, radius_x * 0.5))
        py = int(center_y + np.random.uniform(-radius_y * 0.5, radius_y * 0.5))
        if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
            cv2.circle(img, (px, py), 2, (100, 100, 100), -1)


def generate_samples(output_dir="test_samples"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Clean Good Road (0 potholes)
    road_good = generate_asphalt_texture(800, 600)
    cv2.imwrite(os.path.join(output_dir, "sample_3_good_clean_road.jpg"), road_good)
    print("Created sample_3_good_clean_road.jpg")

    # 2. Moderate Road (2 potholes)
    road_mod = generate_asphalt_texture(800, 600)
    draw_realistic_pothole(road_mod, center_x=280, center_y=380, radius_x=45, radius_y=35, angle=15)
    draw_realistic_pothole(road_mod, center_x=540, center_y=460, radius_x=60, radius_y=42, angle=-20)
    cv2.imwrite(os.path.join(output_dir, "sample_1_moderate_road.jpg"), road_mod)
    print("Created sample_1_moderate_road.jpg")

    # 3. Poor Road (4 distinct potholes)
    road_poor = generate_asphalt_texture(800, 600)
    draw_realistic_pothole(road_poor, center_x=240, center_y=280, radius_x=38, radius_y=30, angle=25)
    draw_realistic_pothole(road_poor, center_x=490, center_y=320, radius_x=50, radius_y=38, angle=-10)
    draw_realistic_pothole(road_poor, center_x=310, center_y=470, radius_x=65, radius_y=48, angle=5)
    draw_realistic_pothole(road_poor, center_x=610, center_y=490, radius_x=55, radius_y=40, angle=-30)
    cv2.imwrite(os.path.join(output_dir, "sample_2_poor_road.jpg"), road_poor)
    print("Created sample_2_poor_road.jpg")


if __name__ == "__main__":
    generate_samples()
