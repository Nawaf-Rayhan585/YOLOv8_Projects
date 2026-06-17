# ==========================
# IMPORTS
# ==========================
from ultralytics import YOLO
import cv2
import time
import os
import csv
from collections import defaultdict

# ==========================
# CONFIGURATION
# ==========================
MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "video/human.mp4"
OUTPUT_VIDEO = "output.mp4"
CONFIDENCE = 0.5

# ==========================
# LOAD MODEL
# ==========================
model = YOLO(MODEL_PATH)

# ==========================
# STATISTICS
# ==========================
total_frames = 0
total_objects = 0
class_counter = defaultdict(int)

# ==========================
# VIDEO SETUP
# ==========================
cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(5))

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# ==========================
# CSV LOGGER
# ==========================
csv_file = open("detections.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
    ["Frame", "Class", "Confidence"]
)

# ==========================
# PROCESS VIDEO
# ==========================
start_time = time.time()

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    total_frames += 1

    results = model.predict(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )

    boxes = results[0].boxes

    for box in boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        class_name = model.names[cls]

        class_counter[class_name] += 1
        total_objects += 1

        csv_writer.writerow([
            total_frames,
            class_name,
            round(conf, 2)
        ])

    annotated = results[0].plot()

    current_time = time.time()
    fps_live = total_frames / (current_time - start_time)

    cv2.putText(
        annotated,
        f"FPS: {fps_live:.2f}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        annotated,
        f"Objects: {len(boxes)}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,0),
        2
    )

    writer.write(annotated)

    cv2.imshow(
        "YOLOv8 Advanced Detector",
        annotated
    )

    if cv2.waitKey(1) == ord("q"):
        break

# ==========================
# CLEANUP
# ==========================
cap.release()
writer.release()
csv_file.close()
cv2.destroyAllWindows()

# ==========================
# FINAL REPORT
# ==========================
print("\n========== REPORT ==========")
print(f"Frames Processed : {total_frames}")
print(f"Objects Detected : {total_objects}")

for cls, count in sorted(class_counter.items()):
    print(f"{cls}: {count}")

print("============================")

# Made with ❤️ by Nawaf Rayhan
# please give a ⭐ to the repository if you found it useful!
