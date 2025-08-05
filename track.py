import cv2
import numpy as np
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt

# ------------------ PARAMETERS ------------------
VIDEO_PATH = "GX010075.MP4"
OUTPUT_VIDEO = "tracked_output.mp4"
PARTICLE_DIAMETER = 11
MINMASS = 150
MAX_ECCENTRICITY = 0.6
SEARCH_RANGE = 7
MEMORY = 3
FPS = 30
DT = 1 / FPS
N_FRAMES = 600

# Scene scale in mm
TARGET_WIDTH = 92
TARGET_HEIGHT = 52
FRAME_WIDTH = 3840
FRAME_HEIGHT = 2160
scale_x = TARGET_WIDTH / FRAME_WIDTH
scale_y = TARGET_HEIGHT / FRAME_HEIGHT

# ------------------ STEP 1: Read video frames ------------------
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
frame_idx = 0

print("🔄 Reading frames...")
while frame_idx < N_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    frames.append((frame, blurred))  # (original, processed)
    frame_idx += 1
cap.release()
print(f"✅ Loaded {len(frames)} frames")

# ------------------ STEP 2: Particle detection ------------------
print("🔍 Detecting particles...")
features = []
for i, (orig, gray) in enumerate(frames):
    f = tp.locate(gray, diameter=PARTICLE_DIAMETER, minmass=MINMASS, invert=False)
    if f is not None:
        f = f[f['ecc'] < MAX_ECCENTRICITY]
        f['frame'] = i
        features.append(f)

all_features = pd.concat(features).reset_index(drop=True)

# ------------------ STEP 3: Link trajectories ------------------
print("🔗 Linking particles...")
linked = tp.link(all_features, search_range=SEARCH_RANGE, memory=MEMORY)
linked = tp.filter_stubs(linked, threshold=3)

# ------------------ STEP 4: Compute velocity ------------------
linked['x_mm'] = linked['x'] * scale_x
linked['y_mm'] = linked['y'] * scale_y
linked['vx_mm'] = linked.groupby('particle')['x_mm'].diff() / DT
linked['vy_mm'] = linked.groupby('particle')['y_mm'].diff() / DT
linked['v_mm'] = np.sqrt(linked['vx_mm']**2 + linked['vy_mm']**2)

# ------------------ STEP 5: Overlay on video ------------------
print("🎥 Writing tracked video...")

# ✅ Reset index to avoid 'frame' ambiguity
linked = linked.reset_index(drop=True)
frame_groups = linked.groupby('frame')

# Setup video writer
H, W, _ = frames[0][0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (W, H))

particle_paths = {}

for frame_num, (orig_frame, _) in enumerate(frames):
    overlay = orig_frame.copy()

    if frame_num in frame_groups.groups:
        frame_data = frame_groups.get_group(frame_num)
        for _, row in frame_data.iterrows():
            pid = int(row['particle'])
            pos = (int(row['x']), int(row['y']))
            particle_paths.setdefault(pid, []).append(pos)

    # Draw trajectories
    for path in particle_paths.values():
        for i in range(1, len(path)):
            cv2.line(overlay, path[i - 1], path[i], (0, 255, 0), 1)

    # Draw current positions
    if frame_num in frame_groups.groups:
        for _, row in frame_groups.get_group(frame_num).iterrows():
            cv2.circle(overlay, (int(row['x']), int(row['y'])), 4, (0, 0, 255), -1)

    out.write(overlay)

out.release()
print(f"🎬 Saved tracked video as: {OUTPUT_VIDEO}")

# ------------------ STEP 6: Save CSV and Plot ------------------
linked.to_csv("ptv_trajectories_filtered.csv", index=False)
print("📄 Saved trajectory data to: ptv_trajectories_filtered.csv")

plt.figure(figsize=(8, 6))
plt.hist(linked["v_mm"].dropna(), bins=30, edgecolor='black')
plt.xlabel("Velocity (mm/s)")
plt.ylabel("Count")
plt.title("Velocity Distribution")
plt.grid(True)
plt.tight_layout()
plt.show()

