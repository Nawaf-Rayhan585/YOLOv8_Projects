import cv2
from cvzone.HandTrackingModule import HandDetector

detector = HandDetector(detectionCon=0.8, maxHands=2)
video = cv2.VideoCapture("hands.mp4")

if not video.isOpened():
    print("Could not open the video file.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        print("End of video reached.")
        break

    h, w = frame.shape[:2]
    hands, processed_frame = detector.findHands(frame)

    if hands:
        for hand in hands:
            hand_type = hand.get("type", "Unknown")
            cx, cy    = hand["center"]

            # fallback if type is Unknown
            if hand_type not in ("Left", "Right"):
                hand_type = "Right" if cx < w // 2 else "Left"

            color = (0, 220, 0) if hand_type == "Right" else (0, 180, 255)
            font  = cv2.FONT_HERSHEY_DUPLEX
            scale, thick = 3.0, 6

            tw, _ = cv2.getTextSize(hand_type, font, scale, thick)[0]
            tx = cx - tw // 2
            ty = cy - 40

            cv2.putText(processed_frame, hand_type, (tx+2, ty+2), font, scale, (0,0,0), thick+2)
            cv2.putText(processed_frame, hand_type, (tx,   ty),   font, scale, color,   thick)

    cv2.imshow("Hand Detection", processed_frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
