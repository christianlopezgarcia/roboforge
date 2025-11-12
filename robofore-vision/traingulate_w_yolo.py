import cv2
import numpy as np
import time
import threading
import queue
from ultralytics import YOLO

# ===============================
# USER PARAMETERS
# ===============================
# Optional correction factor for distance scaling
SCALE_FACTOR = 0.5  # Adjust empirically if distances are consistently off
LEFT_CAM_ID = 0
RIGHT_CAM_ID = 2
BASELINE = 0.05              # meters (distance between cameras)
FOCAL_LENGTH_PX = 700        # pixels (adjust based on calibration)
CONF_THRESH = 0.5
MODEL_PATH = '/home/clopezgarcia2/Desktop/roboforge/robofore-vision/trained_yolo_model/ODM-ver5_ncnn_model'
RESOLUTION = (640, 480)
CAMERA_FRAME_RATE = 30
CAMERA_FOURCC = cv2.VideoWriter_fourcc(*"MJPG")



# ===============================
# CAMERA THREAD CLASS
# ===============================
class CameraThread:
    def __init__(self, cam_id, width, height, fps, fourcc):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.q = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None
        self.black_frame = np.zeros((height, width, 3), dtype=np.uint8)

        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {cam_id} failed to open")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def next(self, black=True, wait=0.01):
        """Returns latest frame or black frame if timeout"""
        try:
            frame = self.q.get(timeout=wait)
        except queue.Empty:
            frame = self.black_frame if black else None
        return frame

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.cap.release()

# ===============================
# TRIANGULATION FUNCTION
# ===============================
def triangulate_point(x_left, x_right, f_px, baseline_m, eps=1e-6):
    disparity = float(x_left) - float(x_right)
    if abs(disparity) < eps:
        return None
    Z = (f_px * baseline_m) / disparity
    X = (Z * x_left) / f_px
    Y = 0.0
    return X, Y, Z

# ===============================
# INITIALIZATION
# ===============================
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
labels = model.names
print("Loaded model with labels:", labels)

print("Starting camera threads...")
camL = CameraThread(LEFT_CAM_ID, RESOLUTION[0], RESOLUTION[1], CAMERA_FRAME_RATE, CAMERA_FOURCC)
camR = CameraThread(RIGHT_CAM_ID, RESOLUTION[0], RESOLUTION[1], CAMERA_FRAME_RATE, CAMERA_FOURCC)
camL.start()
camR.start()
time.sleep(0.2)
print("Cameras initialized and threads running.")

cv2.namedWindow("Stereo YOLO Triangulation", cv2.WINDOW_NORMAL)
cv2.startWindowThread()

fps_buffer = []
fps_avg_len = 50
last_distances = {}  # Store last known distances per object

print("Starting stereo detection... Press 'q' to quit.")

# ===============================
# MAIN LOOP
# ===============================
try:
    while True:
        t0 = time.perf_counter()

        frameL = camL.next()
        frameR = camR.next()
        if frameL is None or frameR is None:
            print("No frames received; continuing.")
            continue

        # YOLO inference
        resultsL = model(frameL, verbose=False)
        resultsR = model(frameR, verbose=False)
        detectionsL = resultsL[0].boxes
        detectionsR = resultsR[0].boxes

        centers_left, centers_right = [], []

        # LEFT detections
        for det in detectionsL:
            conf = float(det.conf)
            if conf < CONF_THRESH:
                continue
            cls_id = int(det.cls)
            name = labels[cls_id]
            xyxy = det.xyxy.cpu().numpy().squeeze()
            xmid = float((xyxy[0] + xyxy[2]) / 2)
            ymid = float((xyxy[1] + xyxy[3]) / 2)
            centers_left.append((name, xmid, ymid))
            cv2.rectangle(frameL, (int(xyxy[0]), int(xyxy[1])),
                          (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(frameL, f"{name}:{conf:.2f}",
                        (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # RIGHT detections
        for det in detectionsR:
            conf = float(det.conf)
            if conf < CONF_THRESH:
                continue
            cls_id = int(det.cls)
            name = labels[cls_id]
            xyxy = det.xyxy.cpu().numpy().squeeze()
            xmid = float((xyxy[0] + xyxy[2]) / 2)
            ymid = float((xyxy[1] + xyxy[3]) / 2)
            centers_right.append((name, xmid, ymid))
            cv2.rectangle(frameR, (int(xyxy[0]), int(xyxy[1])),
                          (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
            cv2.putText(frameR, f"{name}:{conf:.2f}",
                        (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Match detections and triangulate
        y_thresh = 30  # pixels
        for nameL, xL, yL in centers_left:
            match = None
            best_dy = 9999
            for nameR, xR, yR in centers_right:
                if nameL != nameR:
                    continue
                dy = abs(yL - yR)
                if dy < best_dy:
                    best_dy = dy
                    match = (xR, yR)
            if match and best_dy < y_thresh:
                xR, yR = match
                result = triangulate_point(xL, xR, FOCAL_LENGTH_PX, BASELINE)
                if result:
                    X, Y, Z = result
                    Z *= SCALE_FACTOR  # Apply empirical correction
                    distance = np.sqrt(X**2 + Y**2 + Z**2)
                    last_distances[nameL] = distance

                    # Draw distance info
                    text = f"{nameL}: {distance:.2f} m"
                    cv2.putText(frameL, text, (int(xL) - 60, int(yL) - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    cv2.putText(frameR, text, (int(xR) - 60, int(yR) - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    print(f"Last known distance from {nameL}: {distance:.2f} meters")

        # Combine and show FPS
        combined = np.hstack((frameL, frameR))
        fps = 1.0 / (time.perf_counter() - t0)
        fps_buffer.append(fps)
        if len(fps_buffer) > fps_avg_len:
            fps_buffer.pop(0)
        avg_fps = np.mean(fps_buffer)
        cv2.putText(combined, f"FPS: {avg_fps:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Display
        cv2.imshow("Stereo YOLO Triangulation", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('p'):
            cv2.imwrite("capture_stereo.png", combined)
            print("Saved capture_stereo.png")

except KeyboardInterrupt:
    print("Interrupted by user.")

# ===============================
# CLEANUP
# ===============================
print(f"Average FPS: {np.mean(fps_buffer):.2f}")
camL.stop()
camR.stop()
cv2.destroyAllWindows()
