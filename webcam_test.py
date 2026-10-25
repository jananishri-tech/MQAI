

import cv2
import time

# Try opening the default webcam using DirectShow
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Could not open webcam. Trying default camera mode...")
    camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Webcam not detected.")
else:
    print("Webcam opened. Starting video...")

    # Allow the camera time to initialize
    time.sleep(2)

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read a frame. Retrying...")
            time.sleep(0.5)
            continue

        cv2.imshow("Manufacturing Quality AI - Live Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
print("Webcam closed.")
