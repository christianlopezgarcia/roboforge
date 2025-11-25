# ------------------------------
# Notice
# ------------------------------

# Copyright 1966 Clayton Darwin claytondarwin@gmail.com
# Adjusted for YOLO Object Detection and Pinhole Stereo Triangulation
# IMPORTANT: K1 Distortion is now applied correctly to pixel coordinates before triangulation.

# ------------------------------
# Imports
# ------------------------------

import time
import traceback
import math
import threading
import queue
import copy # For safe dictionary access
import sys # for robust logging

import numpy as np
import cv2
from ultralytics import YOLO

# ------------------------------
# Configuration (TUNED FOR K1 CORRECTION)
# ------------------------------
SCALE_FACTOR = 1.0              # Empirical correction on Z. Set to 1.0 to disable.
LEFT_CAM_ID = 0
RIGHT_CAM_ID = 2
BASELINE = 0.051                # meters (5.1 cm baseline)
FOCAL_LENGTH_PX = 422        #  422 -cal 457 - exp
CONF_THRESH = 0.5
MODEL_PATH = '/home/clopezgarcia2/Desktop/roboforge/roboforge_vision/trained_yolo_model/ODM-ver5_ncnn_model'
RESOLUTION = (640, 480)         # (width, height)
CAMERA_FRAME_RATE = 20
CAMERA_FOURCC = cv2.VideoWriter_fourcc(*"MJPG")
Y_THRESH_PX = 40                # Allowed vertical misalignment for valid stereo match
K1_DISTORTION_FACTOR = -0.3    # -.2 to -.3

# ------------------------------
# Global Variables for External Access
# ------------------------------
GLOBAL_TARGET_INFO = {} 
TARGET_INFO_LOCK = threading.Lock()

# New global for saving the last frame
LAST_FRAME_STITCHED = None 
LAST_FRAME_LOCK = threading.Lock()

# ------------------------------
# CAMERA THREAD CLASS (Unchanged)
# ------------------------------
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
        self.current_frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {cam_id} failed to open")
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            if not self.buffer.empty():
                try: self.buffer.get_nowait()
                except queue.Empty: pass
            self.buffer.put(frame)
    def next(self, black=True, wait=0.01):
        try: return self.buffer.get(timeout=wait)
        except queue.Empty: return self.black_frame if black else None
    def stop(self):
        self.running = False
        if self.thread: self.thread.join()
        self.cap.release()

# ------------------------------
# TRIANGULATION UTILITIES (Unchanged, operates on PIXEL coordinates)
# ------------------------------
def triangulate_from_pixel(x_left_px, y_left_px, x_right_px, y_right_px,
                             fx_px, baseline_m, cx_px, cy_px, eps=1e-6):
    """
    Calculates 3D coordinates (X, Y, Z) based on stereo vision pinhole model.
    Note: This function now expects undistorted pixel coordinates as input.
    """
    # Relative pixel coordinates from the image center (cx, cy)
    xl_rel = float(x_left_px) - float(cx_px)
    xr_rel = float(x_right_px) - float(cx_px)
    yl_rel = float(y_left_px) - float(cy_px)
    
    # Disparity calculation
    disparity = xl_rel - xr_rel
    if abs(disparity) < eps: 
        return None # Avoid division by zero/near-zero

    # Z (Depth) calculation
    Z = (fx_px * baseline_m) / disparity
    # X and Y (Lateral and Vertical) calculation
    X = (xl_rel * Z) / fx_px
    Y = (yl_rel * Z) / fx_px
    return float(X), float(Y), float(Z)

# ------------------------------
# DISTORTION CORRECTION UTILITY (NEW)
# ------------------------------
def undistort_pixel_coords(x_d, y_d, cx, cy, fx, k1):
    """
    Applies inverse radial distortion (k1) to a distorted pixel coordinate (x_d, y_d).
    This assumes fx = fy (square pixels).
    """
    # 1. Normalize distorted pixel coordinates (convert to unit distance from center, in focal length units)
    # These are our distorted normalized coordinates (x_bar_d, y_bar_d)
    x_bar_d = (x_d - cx) / fx
    y_bar_d = (y_d - cy) / fx
    
    # 2. Calculate radial distance squared (r^2) in normalized space
    r_sq = x_bar_d**2 + y_bar_d**2
    
    # 3. Calculate the inverse distortion factor (1 / (1 + k1 * r^2))
    # This gives us the undistorted normalized coordinates (x_bar_u, y_bar_u)
    distortion_factor = 1.0 + k1 * r_sq
    x_bar_u = x_bar_d / distortion_factor
    y_bar_u = y_bar_d / distortion_factor
    
    # 4. Denormalize back to undistorted pixel coordinates
    x_u = x_bar_u * fx + cx
    y_u = y_bar_u * fx + cy
    
    return x_u, y_u

# ------------------------------
# Helper for Display (Unchanged)
# ------------------------------
def frame_add_crosshairs(frame, x_c, y_c, size=15, color=(0,255,0), thickness=1):
    """Draws a crosshair at the specified pixel coordinates."""
    if x_c > 0 and y_c > 0:
        x, y = int(x_c), int(y_c)
        cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
        cv2.line(frame, (x, y - size), (x, y + size), color, thickness)

# ------------------------------
# Main Execution Function
# ------------------------------

def run():
    # Declare globals here, inside the function, for modification access
    global GLOBAL_TARGET_INFO
    global TARGET_INFO_LOCK
    global LAST_FRAME_STITCHED
    global LAST_FRAME_LOCK

    try:
        # ------------------------------
        # Setup
        # ------------------------------
        pixel_width, pixel_height = RESOLUTION
        frame_rate = CAMERA_FRAME_RATE
        camera_separation = BASELINE 
        K1 = K1_DISTORTION_FACTOR 
        S_FACTOR = SCALE_FACTOR # Use local variable for scale

        print("Loading YOLO model...")
        model = YOLO(MODEL_PATH)
        labels = model.names

        # Initialize Camera Threads
        ct1 = CameraThread(LEFT_CAM_ID, pixel_width, pixel_height, frame_rate, CAMERA_FOURCC)
        ct2 = CameraThread(RIGHT_CAM_ID, pixel_width, pixel_height, frame_rate, CAMERA_FOURCC)

        ct1.start()
        ct2.start()

        # Camera Intrinsic Parameters
        cx = pixel_width / 2.0
        cy = pixel_height / 2.0
        fx = FOCAL_LENGTH_PX

        time.sleep(0.5)

        # Variables for main screen display
        X, Y, Z, D = 0.0, 0.0, 0.0, 0.0
        x1m, y1m, x2m, y2m = 0.0, 0.0, 0.0, 0.0 
        
        # Logging setup
        last_log_time = time.time()
        log_interval = 5.0 
        
        # ------------------------------
        # Targeting Loop
        # ------------------------------

        while True:
            with TARGET_INFO_LOCK:
                GLOBAL_TARGET_INFO.clear() #previosuly was  empty dict {}
            
            first_match_found = False

            frame1 = ct1.next(black=True, wait=1)
            frame2 = ct2.next(black=True, wait=1)

            # Run YOLO detection and collect centers (distorted pixels)
            resultsL = model(frame1, verbose=False)
            resultsR = model(frame2, verbose=False)
            detsL = resultsL[0].boxes
            detsR = resultsR[0].boxes

            centers_left = []
            for det in detsL:
                # ... (collect left detections and draw boxes) ...
                conf = float(det.conf)
                if conf < CONF_THRESH: continue
                cls_id = int(det.cls)
                name = labels[cls_id]
                xyxy = det.xyxy.cpu().numpy().squeeze()
                xmid = float((xyxy[0] + xyxy[2]) / 2.0)
                ymid = float((xyxy[1] + xyxy[3]) / 2.0)
                centers_left.append({'name': name, 'x': xmid, 'y': ymid, 'conf': conf, 'box': xyxy, 'used': False})
                cv2.rectangle(frame1, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

            centers_right = []
            for det in detsR:
                # ... (collect right detections and draw boxes) ...
                conf = float(det.conf)
                if conf < CONF_THRESH: continue
                cls_id = int(det.cls)
                name = labels[cls_id]
                xyxy = det.xyxy.cpu().numpy().squeeze()
                xmid = float((xyxy[0] + xyxy[2]) / 2.0)
                ymid = float((xyxy[1] + xyxy[3]) / 2.0)
                centers_right.append({'name': name, 'x': xmid, 'y': ymid, 'conf': conf, 'box': xyxy, 'used': False})
                cv2.rectangle(frame2, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
            
            # --- Stereo Matching and Triangulation for ALL objects ---
            for detL in centers_left:
                best_match = None
                best_score = float('inf')
                
                # Find the best corresponding right-side detection
                for detR in centers_right:
                    if detR['used'] or detR['name'] != detL['name']:
                        continue
                    
                    # Epipolar constraint check (vertical alignment) - done on distorted Y
                    dy = abs(detL['y'] - detR['y'])
                    if dy > Y_THRESH_PX: continue
                    
                    score = dy + 0.1 * abs(detL['conf'] - detR['conf'])
                    if score < best_score:
                        best_score = score
                        best_match = detR
                
                if best_match is not None:
                    best_match['used'] = True 
                    
                    # --- K1 DISTORTION CORRECTION: Undistort Pixel Inputs ---
                    x1u, y1u = undistort_pixel_coords(detL['x'], detL['y'], cx, cy, fx, K1)
                    x2u, y2u = undistort_pixel_coords(best_match['x'], best_match['y'], cx, cy, fx, K1)

                    # 1. Triangulate to get 3D coordinates using undistorted pixels
                    tri = triangulate_from_pixel(x1u, y1u, x2u, y2u, 
                                                 fx, camera_separation, cx, cy)
                    
                    if tri is not None:
                        X, Y, Z = tri
                        
                        # Apply the empirical Z scaling factor if it is not 1.0
                        if S_FACTOR != 1.0:
                            X *= S_FACTOR
                            Y *= S_FACTOR
                            Z *= S_FACTOR

                        D = math.sqrt(X*X + Y*Y + Z*Z)
                        avg_conf = (detL['conf'] + best_match['conf']) / 2.0

                        # --- Update Global Dictionary ---
                        obj_key = detL['name']
                        counter = 1
                        
                        with TARGET_INFO_LOCK:
                            while obj_key in GLOBAL_TARGET_INFO:
                                obj_key = f"{detL['name']}_{counter}"
                                counter += 1
                            #print("[Vision] Updating GLOBAL_TARGET_INFO with a detection")
                            GLOBAL_TARGET_INFO[obj_key] = {
                                'X': X, 'Y': Y, 'Z': Z, 'D': D, 
                                'confidence': avg_conf,
                                'ts': time.time()
                            }

                        # --- Update Single Target Display Variables ---
                        if not first_match_found:
                            X, Y, Z, D = X, Y, Z, D
                            # Display crosshairs on the ORIGINAL (distorted) detected position
                            x1m, y1m = detL['x'], detL['y'] 
                            x2m, y2m = best_match['x'], best_match['y']
                            first_match_found = True
                            
                        # Overlay distance on target boxes
                        txt = f"{obj_key}: {D:.2f} m"
                        cv2.putText(frame1, txt, (int(detL['x']) - 40, int(detL['y']) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
                        cv2.putText(frame2, txt, (int(best_match['x']) - 40, int(best_match['y']) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)

            # --- ROBUST 5-SECOND LOGGING ---
            if time.time() - last_log_time >= log_interval:
                with TARGET_INFO_LOCK:
                    targets = copy.deepcopy(GLOBAL_TARGET_INFO)
                
                if targets:
                    print(f"\n--- {time.strftime('%H:%M:%S')} Targets Detected (K1={K1}, X, Y, Z, D in meters) ---")
                    for key, info in targets.items():
                        print(f"  {key:<15} | X:{info['X']:6.2f} Y:{info['Y']:6.2f} Z:{info['Z']:6.2f} D:{info['D']:6.2f} | Conf:{info['confidence']:.2f}")
                    sys.stdout.flush()
                last_log_time = time.time()


            if not first_match_found:
                # Reset display coordinates if no match was found
                X, Y, Z, D = 0.0, 0.0, 0.0, 0.0
                x1m, y1m, x2m, y2m = 0.0, 0.0, 0.0, 0.0

            # Display camera centers and coordinate data for the first target
            frame_add_crosshairs(frame1, cx, cy)
            frame_add_crosshairs(frame2, cx, cy)

            fps1 = int(ct1.current_frame_rate) 
            fps2 = int(ct2.current_frame_rate)
            text = 'X: {:3.2f}m\nY: {:3.2f}m\nZ: {:3.2f}m\nD: {:3.2f}m\nFPS: {}/{}'.format(X,Y,Z,D,fps1,fps2)
            lineloc = 0
            lineheight = 30
            for t in text.split('\n'):
                lineloc += lineheight
                cv2.putText(frame1, t, (10,lineloc), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 1, cv2.LINE_AA, False)

            # Display crosshairs for the first matched target (uses original distorted pixel for screen drawing)
            if x1m != 0.0:
                frame_add_crosshairs(frame1,x1m,y1m,48,color=(0,255,255))
                frame_add_crosshairs(frame2,x2m,y2m,48,color=(0,255,255))

            # Display frame
            cv2.imshow("Left Camera 1 - YOLO",frame1)
            cv2.imshow("Right Camera 2 - YOLO",frame2)

            # Save the stitched frame
            stitched_frame = np.concatenate((frame1, frame2), axis=1)
            with LAST_FRAME_LOCK:
                LAST_FRAME_STITCHED = stitched_frame.copy()

            # Detect keys
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty('Left Camera 1 - YOLO',cv2.WND_PROP_VISIBLE) < 1 or \
               cv2.getWindowProperty('Right Camera 2 - YOLO',cv2.WND_PROP_VISIBLE) < 1 or \
               key == ord('q'):
                break
            elif key != 255:
                print('KEY PRESS:',[chr(key)])

    except:
        print(traceback.format_exc())

    finally:
        # --- IMAGE SAVE ON QUIT ---
        if LAST_FRAME_STITCHED is not None:
            # Add grid for visualization before saving
            for x in range(0, LAST_FRAME_STITCHED.shape[1], 50): 
                cv2.line(LAST_FRAME_STITCHED, (x, 0), (x, LAST_FRAME_STITCHED.shape[0]), (255, 255, 255), 1)
            for y in range(0, LAST_FRAME_STITCHED.shape[0], 50):
                cv2.line(LAST_FRAME_STITCHED, (0, y), (LAST_FRAME_STITCHED.shape[1], y), (255, 255, 255), 1)
                
            filename = f"stereo_capture_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, LAST_FRAME_STITCHED)
            print(f"\nSaved last frame with pixel grid to: {filename}")

        # Close all
        try: ct1.stop() 
        except: pass
        try: ct2.stop() 
        except: pass
        cv2.destroyAllWindows()
        print('DONE')


# ------------------------------
# THREAD SUPPORT FOR MAIN PROGRAM
# ------------------------------

import threading

STEREO_THREAD = None
STOP_FLAG = False

def start_thread():
    """Starts run() inside a daemon thread so main.py can use it."""
    # print("[Vision] loop alive")

    global STEREO_THREAD, STOP_FLAG
    if STEREO_THREAD is not None and STEREO_THREAD.is_alive():
        print("[Stereo] Already running.")
        return

    STOP_FLAG = False

    def runner():
        run()  # your original run() loops until STOP_FLAG or 'q'

    STEREO_THREAD = threading.Thread(target=runner, daemon=True)
    STEREO_THREAD.start()
    print("[Stereo] Started thread.")


def stop_thread():
    """Signals run() to exit cleanly."""
    global STOP_FLAG
    STOP_FLAG = True
    print("[Stereo] Stop requested.")


if __name__ == '__main__':
    run()