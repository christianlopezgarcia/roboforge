# ------------------------------
# Notice
# ------------------------------

# Copyright 1966 Clayton Darwin claytondarwin@gmail.com
# Adjusted for YOLO Object Detection and Pinhole Stereo Triangulation

# ------------------------------
# Imports
# ------------------------------

import time
import traceback
import math
import threading
import queue
import copy # For safe dictionary access

import numpy as np
import cv2
from ultralytics import YOLO

# ------------------------------
# Configuration (Updated BASELINE)
# ------------------------------
SCALE_FACTOR = 1.0          # empirical correction on Z. set to 1.0 to disable.
LEFT_CAM_ID = 0
RIGHT_CAM_ID = 2
BASELINE = 0.051               # meters (Distance between camera centers, set to 5.1 m as requested)
FOCAL_LENGTH_PX = 457       # pixels (keep as your calibrated/assumed fx)
CONF_THRESH = 0.5
MODEL_PATH = '/home/clopezgarcia2/Desktop/roboforge/robofore-vision/trained_yolo_model/ODM-ver5_ncnn_model'
RESOLUTION = (640, 480)     # (width, height)
CAMERA_FRAME_RATE = 20
CAMERA_FOURCC = cv2.VideoWriter_fourcc(*"MJPG")
Y_THRESH_PX = 40            # Allowed vertical misalignment for valid stereo match

# ------------------------------
# Global Variables for External Access
# ------------------------------
# Access this variable from another thread to get the latest coordinates.
# It holds {object_name: {'X': X, 'Y': Y, 'Z': Z, 'D': D, 'confidence': avg_conf, 'ts': timestamp}}
GLOBAL_TARGET_INFO = {} 

# This lock MUST be used whenever reading or writing to GLOBAL_TARGET_INFO
TARGET_INFO_LOCK = threading.Lock()

# ------------------------------
# CAMERA THREAD CLASS (Re-used)
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
# TRIANGULATION UTILITIES (Re-used)
# ------------------------------
def triangulate_from_pixel(x_left_px, y_left_px, x_right_px, y_right_px,
                           fx_px, baseline_m, cx_px, cy_px, eps=1e-6):
    xl_rel = float(x_left_px) - float(cx_px)
    xr_rel = float(x_right_px) - float(cx_px)
    yl_rel = float(y_left_px) - float(cy_px)
    
    disparity = xl_rel - xr_rel
    if abs(disparity) < eps: return None

    Z = (fx_px * baseline_m) / disparity
    X = (xl_rel * Z) / fx_px
    Y = (yl_rel * Z) / fx_px
    return float(X), float(Y), float(Z)

# ------------------------------
# Helper for Display (Re-used)
# ------------------------------
def frame_add_crosshairs(frame, x_c, y_c, size=15, color=(0,255,0), thickness=1):
    if x_c > 0 and y_c > 0:
        x, y = int(x_c), int(y_c)
        cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
        cv2.line(frame, (x, y - size), (x, y + size), color, thickness)

# ------------------------------
# Testing
# ------------------------------

def run():
    # Declare globals here, inside the function, for modification access
    global GLOBAL_TARGET_INFO
    global TARGET_INFO_LOCK

    try:
        # ------------------------------
        # setup
        # ------------------------------
        pixel_width, pixel_height = RESOLUTION
        frame_rate = CAMERA_FRAME_RATE
        camera_separation = BASELINE 

        print("Loading YOLO model...")
        model = YOLO(MODEL_PATH)
        labels = model.names

        ct1 = CameraThread(LEFT_CAM_ID, pixel_width, pixel_height, frame_rate, CAMERA_FOURCC)
        ct2 = CameraThread(RIGHT_CAM_ID, pixel_width, pixel_height, frame_rate, CAMERA_FOURCC)

        ct1.start()
        ct2.start()

        cx = pixel_width / 2.0
        cy = pixel_height / 2.0
        fx = FOCAL_LENGTH_PX

        time.sleep(0.5)

        # Variables for main screen display (will show the first matched item)
        X, Y, Z, D = 0.0, 0.0, 0.0, 0.0
        x1m, y1m, x2m, y2m = 0.0, 0.0, 0.0, 0.0 
        
        # ------------------------------
        # targeting loop
        # ------------------------------

        while 1:
            # Clear the global dictionary safely at the start of the frame processing
            with TARGET_INFO_LOCK:
                GLOBAL_TARGET_INFO = {} 
            
            first_match_found = False # Flag to track the first successful match for screen overlay

            frame1 = ct1.next(black=True, wait=1)
            frame2 = ct2.next(black=True, wait=1)

            # Run YOLO and parse detections
            resultsL = model(frame1, verbose=False)
            resultsR = model(frame2, verbose=False)
            detsL = resultsL[0].boxes
            detsR = resultsR[0].boxes

            centers_left = []
            # Collect and draw Left detections
            for det in detsL:
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
            # Collect and draw Right detections
            for det in detsR:
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
                
                for detR in centers_right:
                    if detR['used'] or detR['name'] != detL['name']:
                        continue
                    
                    dy = abs(detL['y'] - detR['y'])
                    if dy > Y_THRESH_PX: continue
                    
                    # Score favors smallest vertical difference
                    score = dy + 0.1 * abs(detL['conf'] - detR['conf'])
                    if score < best_score:
                        best_score = score
                        best_match = detR
                
                if best_match is not None:
                    # Mark right detection as used to prevent double-matching
                    best_match['used'] = True 
                    
                    # Triangulate
                    tri = triangulate_from_pixel(detL['x'], detL['y'], best_match['x'], best_match['y'], 
                                                       fx, camera_separation, cx, cy)
                    
                    if tri is not None:
                        X_raw, Y_raw, Z_raw = tri
                        Z = Z_raw * SCALE_FACTOR
                        X = ( (detL['x'] - cx) * Z ) / fx
                        Y = ( (detL['y'] - cy) * Z ) / fx
                        D = math.sqrt(X*X + Y*Y + Z*Z)
                        avg_conf = (detL['conf'] + best_match['conf']) / 2.0

                        # --- 1. Update Global Dictionary (Protected by Lock) ---
                        obj_key = detL['name']
                        counter = 1
                        
                        # Find a unique key safely
                        with TARGET_INFO_LOCK:
                            while obj_key in GLOBAL_TARGET_INFO:
                                obj_key = f"{detL['name']}_{counter}"
                                counter += 1

                        # Write the final entry safely
                        with TARGET_INFO_LOCK:
                            GLOBAL_TARGET_INFO[obj_key] = {
                                'X': X, 'Y': Y, 'Z': Z, 'D': D, 
                                'confidence': avg_conf,
                                'ts': time.time()
                            }

                        # --- 2. Update Single Target Display Variables (for legacy output) ---
                        if not first_match_found:
                            X, Y, Z, D = X, Y, Z, D
                            x1m, y1m = detL['x'], detL['y']
                            x2m, y2m = best_match['x'], best_match['y']
                            first_match_found = True # Lock to the first object found in this frame
                            
                        # Overlay distance on target boxes
                        txt = f"{obj_key}: {D:.2f} m"
                        cv2.putText(frame1, txt, (int(detL['x']) - 40, int(detL['y']) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
                        cv2.putText(frame2, txt, (int(best_match['x']) - 40, int(best_match['y']) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)


            if not first_match_found:
                # Reset display coordinates if no match was found
                X, Y, Z, D = 0.0, 0.0, 0.0, 0.0
                x1m, y1m, x2m, y2m = 0.0, 0.0, 0.0, 0.0

            # display camera centers and coordinate data for the first target
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

            # display crosshairs for the first matched target
            if x1m != 0.0:
                frame_add_crosshairs(frame1,x1m,y1m,48,color=(0,255,255))
                frame_add_crosshairs(frame2,x2m,y2m,48,color=(0,255,255))

            # display frame
            cv2.imshow("Left Camera 1 - YOLO",frame1)
            cv2.imshow("Right Camera 2 - YOLO",frame2)

            # detect keys
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
        # close all
        try: ct1.stop() 
        except: pass
        try: ct2.stop() 
        except: pass
        cv2.destroyAllWindows()
        print('DONE')

if __name__ == '__main__':
    run()