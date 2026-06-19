from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO("yolo11n.pt")  # or yolo11s.pt

# Video
cap = cv2.VideoCapture("people.mp4")

# Heatmap accumulator
heatmap = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    if heatmap is None:
        heatmap = np.zeros((h, w), dtype=np.float32)

    # YOLO Detection
    results = model(frame, verbose=False)

    people_count = 0

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])

            # COCO person class
            if cls == 0:
                people_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

                # Center point
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                cv2.circle(frame, (cx, cy), 4,
                           (0, 0, 255), -1)

                # Add heat
                cv2.circle(
                    heatmap,
                    (cx, cy),
                    30,
                    1,
                    -1
                )

    # Smooth heatmap
    heat_blur = cv2.GaussianBlur(
        heatmap,
        (0, 0),
        15
    )

    # Normalize
    heat_norm = cv2.normalize(
        heat_blur,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Color heatmap
    heat_color = cv2.applyColorMap(
        heat_norm,
        cv2.COLORMAP_JET
    )

    # Overlay
    output = cv2.addWeighted(
        frame,
        0.7,
        heat_color,
        0.3,
        0
    )

    # Count text
    cv2.putText(
        output,
        f"People: {people_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("People Flow + Heatmap", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
