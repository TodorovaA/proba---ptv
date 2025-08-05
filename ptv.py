import cv2
import numpy as np
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt

# ---- PARAMETERS ----
VIDEO_PATH = "GX010075.MP4"
PARTICLE_DIAMETER = 11
SEARCH_RANGE = 7
MINMASS = 100
MEMORY = 3
FPS = 30  # frames per second
DT = 1 / FPS

# ---- PHYSICAL DIMENSIONS ----
SOURCE = np.array([
    [0, 0],
    [3840, 0],
    [3840, 2160],
    [0, 2160]
], dtype=np.float32)

TARGET_WIDTH = 92     # mm
TARGET_HEIGHT = 52    # mm

# Scaling factors (pixel → mm)
scale_x = TARGET_WIDTH / 3840
scale_y = TARGET_HEIGHT / 2160

# ---- STEP 1: Load video frames ----
print("Reading selected frames...")
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
target_range = range(400, 420)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret or frame_idx > max(target_range):
        break
    if frame_idx in target_range:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        frames.append(blurred)
    frame_idx += 1

cap.release()
print(f"Loaded {len(frames)} frames")

# ---- STEP 2: Detect particles ----
print("Detecting particles...")
features_list = []
for i, frame in enumerate(frames, start=min(target_range)):
    f = tp.locate(frame, diameter=PARTICLE_DIAMETER, minmass=MINMASS, invert=False)
    f['frame'] = i
    features_list.append(f)

all_features = pd.concat(features_list).reset_index(drop=True)

# ---- STEP 3: Link particles into trajectories ----
print("Linking trajectories...")
tracks = tp.link(all_features, search_range=SEARCH_RANGE, memory=MEMORY)

# ---- STEP 4: Convert to mm and compute velocities ----
print("Computing velocities...")

# Convert positions to mm
tracks['x_mm'] = tracks['x'] * scale_x
tracks['y_mm'] = tracks['y'] * scale_y

# Compute velocities in mm/s
tracks['vx_mm'] = tracks.groupby('particle')['x_mm'].diff() / DT
tracks['vy_mm'] = tracks.groupby('particle')['y_mm'].diff() / DT
tracks['v_mm']  = np.sqrt(tracks['vx_mm']**2 + tracks['vy_mm']**2)

# ---- STEP 5: Save output ----
tracks.to_csv("ptv_tracks_with_velocity_mm.csv", index=False)
print("Results saved to ptv_tracks_with_velocity_mm.csv")

# ---- STEP 6: Plot velocity histogram ----
plt.figure(figsize=(8, 6))
plt.hist(tracks['v_mm'].dropna(), bins=30, edgecolor='black')
plt.xlabel("Velocity (mm/s)")
plt.ylabel("Particle Count")
plt.title("Velocity Distribution")
plt.grid(True)
plt.show()
