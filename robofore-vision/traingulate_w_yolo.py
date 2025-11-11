import cv2
import numpy as np
import time
from ultralytics import YOLO

# ===============================
# USER PARAMETERS
# ===============================
LEFT_CAM_ID = 0
RIGHT_CAM_ID = 2
BASELINE = 0.06  # distance between cameras in meters
FOCAL_LENGTH_PX = 700  # adjust for your camera (f in pixels)
CONF_THRESH = 0.5
MODEL_PATH = '/home/clopezgarcia2/Desktop/roboforge/robofore-vision/trained_yolo_model/ODM-ver5_ncnn_model'
RESOLUTION = (640, 480)

# ===============================
# SETUP
# ===============================
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
labels = model.names
print("Loaded model with labels:", labels)

# Initialize both cameras
cap_left = cv2.VideoCapture(LEFT_CAM_ID)
cap_right = cv2.VideoCapture(RIGHT_CAM_ID)

cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

if not cap_left.isOpened() or not cap_right.isOpened():
    print("Error: One or both cameras not available.")
    exit()

print("Stereo cameras initialized.")


# ===============================
# TRIANGULATION FUNCTION
# ===============================
def triangulate_point(x_left, x_right, f, B):
    disparity = x_left - x_right
    if disparity == 0:
        return None
    Z = (f * B) / disparity
    X = (Z * x_left) / f
    Y = 0  # assuming level ground
    return X, Y, Z


# ===============================
# MAIN LOOP
# ===============================
fps_buf = []
while True:
    t0 = time.time()

    # --- Grab frames ---
    retL, frameL = cap_left.read()
    retR, frameR = cap_right.read()
    if not retL or not retR:
        print("Frame grab failed, check cameras.")
        break

    # --- Run YOLO on each frame ---
    resultsL = model(frameL, verbose=False)
    resultsR = model(frameR, verbose=False)

    detectionsL = resultsL[0].boxes
    detectionsR = resultsR[0].boxes

    centers_left = []
    centers_right = []

    # --- Parse left detections ---
    for det in detectionsL:
        conf = det.conf.item()
        if conf < CONF_THRESH:
            continue
        cls_id = int(det.cls.item())
        name = labels[cls_id]
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmid = int((xyxy[0] + xyxy[2]) / 2)
        ymid = int((xyxy[1] + xyxy[3]) / 2)
        centers_left.append((name, xmid, ymid))
        color = (0, 255, 0)
        cv2.rectangle(frameL, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
        cv2.putText(frameL, f"{name}", (xyxy[0], xyxy[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- Parse right detections ---
    for det in detectionsR:
        conf = det.conf.item()
        if conf < CONF_THRESH:
            continue
        cls_id = int(det.cls.item())
        name = labels[cls_id]
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmid = int((xyxy[0] + xyxy[2]) / 2)
        ymid = int((xyxy[1] + xyxy[3]) / 2)
        centers_right.append((name, xmid, ymid))
        color = (255, 0, 0)
        cv2.rectangle(frameR, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
        cv2.putText(frameR, f"{name}", (xyxy[0], xyxy[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- Triangulate matching color blocks ---
    for nameL, xL, yL in centers_left:
        for nameR, xR, yR in centers_right:
            if nameL == nameR:  # same color label
                result = triangulate_point(xL, xR, FOCAL_LENGTH_PX, BASELINE)
                if result:
                    X, Y, Z = result
                    cv2.putText(frameL, f"{nameL} Dist: {Z:.2f}m",
                                (xL - 40, yL - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frameR, f"{nameR} Dist: {Z:.2f}m",
                                (xR - 40, yR - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # --- Display both frames side-by-side ---
    combined = np.hstack((frameL, frameR))
    fps = 1.0 / (time.time() - t0)
    fps_buf.append(fps)
    if len(fps_buf) > 30:
        fps_buf.pop(0)
    avg_fps = np.mean(fps_buf)
    cv2.putText(combined, f"FPS: {avg_fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Stereo YOLO Triangulation", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
print("Exiting cleanly.")
