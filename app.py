"""Pinch to grab a red ball. Run: python3 app.py"""

import math
import os
import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

hands = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.35,
    min_tracking_confidence=0.35,
))
cam = cv2.VideoCapture(int(os.environ.get("AWARE_CAMERA", "0")))

# Request a higher resolution from the webcam (e.g., 1920x1080 or 1280x720)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Allow the window to be resized
cv2.namedWindow("app", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

bx = by = None
br, grabbed, t = 55, False, 0

while True:
    ok, frame = cam.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    if bx is None:
        bx, by = w // 2, h // 2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t)
    t += 33

    if res.hand_landmarks:
        lm = res.hand_landmarks[0]

        def pt(i):
            return int(lm[i].x * w), int(lm[i].y * h)

        x4, y4 = pt(4)
        x8, y8 = pt(8)
        x0, y0 = pt(0)
        x9, y9 = pt(9)
        hand = max(math.hypot(x9 - x0, y9 - y0), 1)
        tip = math.hypot(x4 - x8, y4 - y8)
        # Relative to hand size so it works at any distance / resolution
        pinch_on = tip < hand * 0.55
        pinch_off = tip > hand * 0.70
        mx, my = (x4 + x8) // 2, (y4 + y8) // 2
        near = math.hypot(mx - bx, my - by) < br + 50

        if grabbed:
            if pinch_off:
                grabbed = False
            else:
                bx, by = mx, my
        elif pinch_on and near:
            grabbed = True
            bx, by = mx, my

        color = (0, 255, 0) if (grabbed or pinch_on) else (255, 200, 0)
        cv2.line(frame, (x4, y4), (x8, y8), color, 2)
        cv2.circle(frame, (x4, y4), 8, color, -1)
        cv2.circle(frame, (x8, y8), 8, color, -1)

    cv2.circle(frame, (bx, by), br, (0, 0, 255), -1)

    frame_4k = cv2.resize(frame, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
        
    cv2.imshow("app", frame_4k)
    # cv2.imshow("app", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
hands.close()
cv2.destroyAllWindows()
