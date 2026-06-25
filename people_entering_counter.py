from ultralytics import YOLO
import cv2
import numpy as np
import supervision as sv

# Load model
model = YOLO("yolov8n.pt")

# Video
cap = cv2.VideoCapture("people.mp4")

# Tracker
tracker = sv.ByteTrack()

# Line position
LINE_Y = 300

up_count = 0
down_count = 0

# Store previous positions
track_history = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[0], verbose=False)[0]

    detections = sv.Detections.from_ultralytics(results)

    detections = tracker.update_with_detections(detections)

    cv2.line(
        frame,
        (0, LINE_Y),
        (frame.shape[1], LINE_Y),
        (0, 0, 255),
        2
    )

    for bbox, track_id in zip(
        detections.xyxy,
        detections.tracker_id
    ):
        if track_id is None:
            continue

        x1, y1, x2, y2 = bbox.astype(int)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if track_id not in track_history:
            track_history[track_id] = cy

        prev_y = track_history[track_id]

        # Moving Down
        if prev_y < LINE_Y and cy >= LINE_Y:
            down_count += 1

        # Moving Up
        elif prev_y > LINE_Y and cy <= LINE_Y:
            up_count += 1

        track_history[track_id] = cy

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )

    # Create dashboard
    dashboard = np.ones((800, 1000, 3), dtype=np.uint8) * 240
    # Header
    cv2.rectangle(dashboard, (0, 0), (1000, 80), (0, 120, 0), -1)
    cv2.putText(
        dashboard,
        "PEOPLE COUNTING",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (255, 255, 255),
        4
    )
    # Resize frame to fit dashboard
    frame_resized = cv2.resize(frame, (900, 500))
    # Put video inside dashboard
    dashboard[90:590, 50:950] = frame_resized
    # OUT Panel
    cv2.rectangle(
        dashboard,
        (50, 620),
        (450, 780),
        (0, 0, 180),
        -1
    )
    cv2.putText(
        dashboard,
        "OUT",
        (70, 715),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        4
    )
    cv2.putText(
        dashboard,
        str(up_count),
        (270, 730),
        cv2.FONT_HERSHEY_DUPLEX,
        3,
        (255, 255, 255),
        6
    )
    # IN Panel
    cv2.rectangle(
        dashboard,
        (550, 620),
        (950, 780),
        (0, 120, 0),
        -1
    )
    cv2.putText(
        dashboard,
        "IN",
        (600, 715),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        4
    )
    cv2.putText(
        dashboard,
        str(down_count),
        (760, 730),
        cv2.FONT_HERSHEY_DUPLEX,
        3,
        (255, 255, 255),
        6
    )
    cv2.imshow("People Counting Dashboard", dashboard)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
