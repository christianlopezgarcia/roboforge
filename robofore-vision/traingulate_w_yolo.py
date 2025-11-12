import cv2
import numpy as np
import time
import threading
import queue
from ultralytics import YOLO
import math

# ===============================
# USER PARAMETERS
# ===============================
SCALE_FACTOR = 1            # empirical correction on Z. set to 1.0 to disable.
LEFT_CAM_ID = 0
RIGHT_CAM_ID = 2
BASELINE = 0.05               # meters (distance between camera centers)
FOCAL_LENGTH_PX = 457#457         # pixels (keep as your calibrated/assumed fx)
CONF_THRESH = 0.5
MODEL_PATH = '/home/clopezgarcia2/Desktop/roboforge/robofore-vision/trained_yolo_model/ODM-ver5_ncnn_model'
RESOLUTION = (640, 480)       # (width, height)
CAMERA_FRAME_RATE = 30
CAMERA_FOURCC = cv2.VideoWriter_fourcc(*"MJPG")

# ===============================
# CAMERA THREAD CLASS (queue maxsize=1)
# ===============================
class CameraThread:
    def __init__(self, cam_id, width, height, fps, fourcc):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.buffer = queue.Queue(maxsize=1)
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
            # keep only newest
            if not self.buffer.empty():
                try:
                    self.buffer.get_nowait()
                except queue.Empty:
                    pass
            self.buffer.put(frame)

    def next(self, black=True, wait=0.01):
        try:
            return self.buffer.get(timeout=wait)
        except queue.Empty:
            return self.black_frame if black else None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.cap.release()

# ===============================
# TRIANGULATION UTILITIES
# ===============================
def triangulate_from_pixel(x_left_px, y_left_px, x_right_px, y_right_px,
                           fx_px, baseline_m, cx_px, cy_px, eps=1e-6):
    """
    Compute (X, Y, Z) in camera coordinate system (left camera center origin).
    Uses pinhole model:
      disparity = xl_rel - xr_rel
      Z = (fx * B) / disparity
      X = (xl_rel * Z) / fx
      Y = (yl_rel * Z) / fy  (use fy=fx if unknown)
    Inputs in pixels. Outputs in meters.
    Returns None if invalid (tiny disparity).
    """
    xl_rel = float(x_left_px) - float(cx_px)
    xr_rel = float(x_right_px) - float(cx_px)   # assume same cx for both cameras (approx)
    yl_rel = float(y_left_px) - float(cy_px)
    yr_rel = float(y_right_px) - float(cy_px)

    disparity = xl_rel - xr_rel
    if abs(disparity) < eps:
        return None

    Z = (fx_px * baseline_m) / disparity
    X = (xl_rel * Z) / fx_px
    # assume square pixels: fy ≈ fx
    Y = (yl_rel * Z) / fx_px
    return float(X), float(Y), float(Z)

def compute_bearing_deg(X, Z):
    # angle (deg) from camera center: positive = right, negative = left
    if Z == 0:
        return 0.0
    return math.degrees(math.atan2(X, Z))

# ===============================
# INITIALIZE MODEL + CAMERAS
# ===============================
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
labels = model.names
print("Model labels:", labels)

print("Starting camera threads...")
camL = CameraThread(LEFT_CAM_ID, RESOLUTION[0], RESOLUTION[1], CAMERA_FRAME_RATE, CAMERA_FOURCC)
camR = CameraThread(RIGHT_CAM_ID, RESOLUTION[0], RESOLUTION[1], CAMERA_FRAME_RATE, CAMERA_FOURCC)
camL.start()
camR.start()
time.sleep(0.2)
print("Cameras running.")

# cv window
cv2.namedWindow("Stereo YOLO Triangulation", cv2.WINDOW_NORMAL)
cv2.startWindowThread()

# bookkeeping
fps_buffer = []
fps_avg_len = 50
last_info = {}   # dict color -> {X,Y,Z,dist,angle,conf,ts}

# precompute principal point
cx = RESOLUTION[0] / 2.0
cy = RESOLUTION[1] / 2.0
fx = FOCAL_LENGTH_PX

print("Starting main loop. Press 'q' to quit.")

try:
    while True:
        loop_t0 = time.perf_counter()

        # grab latest frames (non-blocking fallback to black)
        frameL = camL.next(black=True, wait=0.02)
        frameR = camR.next(black=True, wait=0.02)
        if frameL is None or frameR is None:
            # should not happen because black fallback is True, but just in case
            continue

        # run YOLO on each frame
        resultsL = model(frameL, verbose=False)
        resultsR = model(frameR, verbose=False)
        detsL = resultsL[0].boxes
        detsR = resultsR[0].boxes

        centers_left = []
        centers_right = []

        # parse left detections
        for det in detsL:
            conf = float(det.conf)
            if conf < CONF_THRESH:
                continue
            cls_id = int(det.cls)
            name = labels[cls_id]
            xyxy = det.xyxy.cpu().numpy().squeeze()
            xmid = float((xyxy[0] + xyxy[2]) / 2.0)
            ymid = float((xyxy[1] + xyxy[3]) / 2.0)
            centers_left.append((name, xmid, ymid, conf, tuple(xyxy)))
            cv2.rectangle(frameL, (int(xyxy[0]), int(xyxy[1])),
                          (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(frameL, f"{name}:{conf:.2f}", (int(xyxy[0]), int(xyxy[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # parse right detections
        for det in detsR:
            conf = float(det.conf)
            if conf < CONF_THRESH:
                continue
            cls_id = int(det.cls)
            name = labels[cls_id]
            xyxy = det.xyxy.cpu().numpy().squeeze()
            xmid = float((xyxy[0] + xyxy[2]) / 2.0)
            ymid = float((xyxy[1] + xyxy[3]) / 2.0)
            centers_right.append((name, xmid, ymid, conf, tuple(xyxy)))
            cv2.rectangle(frameR, (int(xyxy[0]), int(xyxy[1])),
                          (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
            cv2.putText(frameR, f"{name}:{conf:.2f}", (int(xyxy[0]), int(xyxy[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        # match left->right: for every left detection find right candidate of same class with min vertical diff
        y_thresh_px = 40  # allowed vertical misalignment in px for valid stereo match (tuneable)
        for nameL, xL, yL, confL, boxL in centers_left:
            best = None
            best_score = float('inf')
            for nameR, xR, yR, confR, boxR in centers_right:
                if nameR != nameL:
                    continue
                dy = abs(yL - yR)
                if dy > y_thresh_px:
                    continue
                # prefer smaller vertical diff and similar confidence
                score = dy + 0.1 * abs(confL - confR)
                if score < best_score:
                    best_score = score
                    best = (xR, yR, confR, boxR)
            if best is None:
                continue
            xR, yR, confR, boxR = best

            # triangulate using principal point correction
            tri = triangulate_from_pixel(xL, yL, xR, yR, fx, BASELINE, cx, cy, eps=1e-6)
            if tri is None:
                # invalid disparity
                continue
            X, Y, Z_raw = tri
            # apply empirical correction if desired
            Z = Z_raw * SCALE_FACTOR if SCALE_FACTOR is not None else Z_raw
            # compute metric X,Y using corrected Z (recompute X,Y from pixel relations to reduce small mismatch)
            X = ( (xL - cx) * Z ) / fx
            Y = ( (yL - cy) * Z ) / fx

            distance = math.sqrt(X*X + Y*Y + Z*Z)
            angle_deg = compute_bearing_deg(X, Z)

            # store last-known info per class (overwrite oldest)
            last_info[nameL] = {
                'X': X, 'Y': Y, 'Z': Z, 'distance': distance,
                'angle_deg': angle_deg, 'confidence': (confL + confR)/2.0,
                'timestamp': time.time()
            }

            # overlay near the detected boxes
            txt = f"{nameL} {distance:.2f}m"
            cv2.putText(frameL, txt, (int(xL) - 40, int(yL) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
            cv2.putText(frameR, txt, (int(xR) - 40, int(yR) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)

        # Build the info panel (left side overlay)
        panel_w = 320
        panel_h = 200
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8) + 30  # dark background
        cv2.rectangle(panel, (0,0), (panel_w-1, panel_h-1), (60,60,60), 1)

        # Title
        cv2.putText(panel, "Last Known Distances", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        # iterate known colors/names in deterministic order (if present)
        y0 = 45
        line_h = 28
        for i, name in enumerate(sorted(last_info.keys())):
            info = last_info[name]
            X = info['X']; Y = info['Y']; Z = info['Z']
            dist = info['distance']; ang = info['angle_deg']; conf = info['confidence']
            ts = info['timestamp']
            txt1 = f"{name}: {dist:.2f} m  ang:{ang:+.1f}°"
            txt2 = f" X:{X:+.2f} Y:{Y:+.2f} Z:{Z:+.2f} c:{conf:.2f}"
            cv2.putText(panel, txt1, (8, y0 + i*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)
            cv2.putText(panel, txt2, (8, y0 + i*line_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)

        # merge panel into left side of combined display (overlay)
        # we will place panel in top-left corner of combined image
        combined = np.hstack((frameL, frameR))
        # ensure panel fits
        h_comb, w_comb = combined.shape[:2]
        if panel_h < h_comb and panel_w < w_comb:
            combined[5:5+panel_h, 5:5+panel_w] = panel

        # fps
        fps = 1.0 / (time.perf_counter() - loop_t0)
        fps_buffer.append(fps)
        if len(fps_buffer) > fps_avg_len:
            fps_buffer.pop(0)
        avg_fps = sum(fps_buffer) / len(fps_buffer)
        cv2.putText(combined, f"FPS: {avg_fps:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # show
        cv2.imshow("Stereo YOLO Triangulation", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.imwrite("capture_stereo.png", combined)
            print("Saved capture_stereo.png")

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    camL.stop()
    camR.stop()
    cv2.destroyAllWindows()
