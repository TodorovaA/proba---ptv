import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import defaultdict

# ------------------ PARAMETERS ------------------
VIDEO_PATH = "GX010075.MP4"
OUTPUT_VIDEO = "tracked_output_vertical_line_counter.mp4"
N_FRAMES = 600
FPS = 30
DT = 1 / FPS
MAX_DISTANCE = 80

MIN_RADIUS = 10
MAX_RADIUS = 2000
AREA_THRESHOLD = 3000

# Scene scale in mm
TARGET_WIDTH = 920
TARGET_HEIGHT = 520
FRAME_WIDTH = 3840
FRAME_HEIGHT = 2160
scale_x = TARGET_WIDTH / FRAME_WIDTH
scale_y = TARGET_HEIGHT / FRAME_HEIGHT

# HSV Color Ranges
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 70, 50])
upper_red2 = np.array([180, 255, 255])
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

# Tracking setup
next_particle_id = 0
active_particles = {}  # id: (x, y, color)
trajectories = defaultdict(list)
positions = []
counted_red = set()
counted_blue = set()
crossed_center_x = {}

# Video processing
cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None
frame_idx = 0

print("🔄 Processing video frames...")

while frame_idx < N_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    if out is None:
        H, W, _ = frame.shape
        center_line_x = W // 2
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (W, H))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

    detections = []

    def process_mask(mask, color_label):
        local_detections = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < AREA_THRESHOLD:
                continue
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius < MIN_RADIUS or radius > MAX_RADIUS:
                continue
            local_detections.append((int(x), int(y), color_label))
        return local_detections

    detections += process_mask(mask_red, "red")
    detections += process_mask(mask_blue, "blue")

    # Tracking assignment
    updated_particles = {}
    used_ids = set()
    for x, y, color in detections:
        matched = False
        for pid, (px, py, pcolor) in active_particles.items():
            if pid in used_ids or color != pcolor:
                continue
            if distance.euclidean((x, y), (px, py)) < MAX_DISTANCE:
                updated_particles[pid] = (x, y, color)
                used_ids.add(pid)
                matched = True
                break
        if not matched:
            updated_particles[next_particle_id] = (x, y, color)
            next_particle_id += 1

    active_particles = updated_particles

    # Draw vertical center line
    cv2.line(frame, (center_line_x, 0), (center_line_x, H), (0, 255, 255), 2)

    # Draw detections, save positions
    for pid, (x, y, color) in active_particles.items():
        col_bgr = (0, 0, 255) if color == "red" else (255, 0, 0)
        cv2.circle(frame, (x, y), 10, col_bgr, -1)
        cv2.putText(frame, str(pid), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        trajectories[pid].append((x, y))

        positions.append({
            "frame": frame_idx,
            "id": pid,
            "x": x,
            "y": y,
            "x_mm": x * scale_x,
            "y_mm": y * scale_y,
            "color": color
        })

        # Line crossing (vertical)
        if pid in crossed_center_x:
            last_x = crossed_center_x[pid]
            if last_x < center_line_x <= x or last_x > center_line_x >= x:
                if color == "red" and pid not in counted_red:
                    counted_red.add(pid)
                elif color == "blue" and pid not in counted_blue:
                    counted_blue.add(pid)
        crossed_center_x[pid] = x

    # Draw trajectories
    for path in trajectories.values():
        if len(path) >= 2:
            for i in range(1, len(path)):
                cv2.line(frame, path[i - 1], path[i], (0, 255, 0), 2)

    # Show counts
    cv2.putText(frame, f"Red Count: {len(counted_red)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Blue Count: {len(counted_blue)}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"🎬 Done! Saved video as: {OUTPUT_VIDEO}")

# Save CSV
df = pd.DataFrame(positions)
df['vx_mm'] = df.groupby('id')['x_mm'].diff() / DT
df['vy_mm'] = df.groupby('id')['y_mm'].diff() / DT
df['v_mm'] = np.sqrt(df['vx_mm']**2 + df['vy_mm']**2)
df.to_csv("particle_tracking_vertical_line.csv", index=False)
print("📄 Saved CSV: particle_tracking_vertical_line.csv")

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(df['v_mm'].dropna(), bins=30, edgecolor='black')
plt.xlabel("Velocity (mm/s)")
plt.ylabel("Count")
plt.title("Velocity Distribution")
plt.grid(True)
plt.tight_layout()
plt.show()
