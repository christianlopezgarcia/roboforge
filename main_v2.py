#!/usr/bin/env python3
"""
main.py — Robot control loop (vision + X-align + ultrasonic + pickup)

Flow:
1. Start vision + ultrasonic threads
2. Detect blocks
3. Pick target (currently CLOSEST by D)
4. Center in X
5. Approach using ultrasonic
6. Grip and place
"""

import time
import busio
import board
from typing import Dict, Any, Optional

from roboforge_arm_v2.armbackup_1117 import RobotArm
from roboforge_motors.MotorControlv2 import myMotors, BNO055, PCA9685PWM

import roboforge_vision.traingulate_w_yolo as vision

from Ultrasonic import (
    start_thread as start_ultra_thread,
    stop_thread as stop_ultra_thread,
    GLOBAL_ULTRASONIC_INFO,
    ULTRASONIC_INFO_LOCK
)

# ---------------------------
# CONFIG
# ---------------------------
I2C_BUS = (board.SCL, board.SDA)

ANGLE_PID_P = 0.2
ANGLE_PID_I = 0.1

X_TOL = 0.03                # meters
ULTRA_TOL_CM = 0.25
VISION_CLOSE_Z = 0.30

LR_STEP_DURATION = 1
MOTOR_PULSE = 0.04
PAUSE_WAIT = 0.2

ULTRA_RATE = 100
ULTRA_ECHO = 17
ULTRA_TRIGGER = 27

# ---------------------------
# HARDWARE INIT
# ---------------------------
i2c = busio.I2C(*I2C_BUS)

arm = RobotArm(i2c)
bno = BNO055(i2c)
pca = PCA9685PWM(i2c)
motors = myMotors(bno, pca, [ANGLE_PID_P, ANGLE_PID_I])

motors.set_pid_enable(1)
motors.move("STP")
motors.turn(0)
motors.update_motors()

time.sleep(0.5)

# ---------------------------
# PICKUP ROUTINES
# ---------------------------
def five_cm_pickup():
    arm.move_to_pose("5cm", reverse=True)
    time.sleep(1)
    arm.move_to_pose("5cm_closed")
    time.sleep(1)
    arm.move_to_pose("5cm_30r_closed")
    time.sleep(1)
    arm.move_to_pose("default_closed", reverse=True)
    time.sleep(1)
    arm.move_to_pose("fold_over", reverse=True)
    time.sleep(1)
    arm.move_to_pose("fold_over_open")
    time.sleep(1)
    arm.move_to_pose("4_21_view", reverse=True)

def eight_cm_pickup():
    arm.move_to_pose("8cm", reverse=True)
    time.sleep(1)
    arm.move_to_pose("8cm_closed")
    time.sleep(1)
    arm.move_to_pose("8cm_30r_closed")
    time.sleep(1)
    arm.move_to_pose("fold_over", reverse=True)
    time.sleep(1)
    arm.move_to_pose("fold_over_open")
    time.sleep(1)
    arm.move_to_pose("4_21_view", reverse=True)

def ten_cm_pickup():
    arm.move_to_pose("10_cm", reverse=True)
    time.sleep(1)
    arm.move_to_pose("10_cm_closed")
    time.sleep(1)
    arm.move_to_pose("10_cm_30r_closed")
    time.sleep(1)
    arm.move_to_pose("fold_over", reverse=True)
    time.sleep(1)
    arm.move_to_pose("fold_over_open")
    time.sleep(1)
    arm.move_to_pose("4_21_view", reverse=True)

DISTANCE_TO_ACTION_MAP = {
    5: five_cm_pickup,
    8: eight_cm_pickup,
    10: ten_cm_pickup
}

# ---------------------------
# HELPERS
# ---------------------------
unique_blocks = []

def get_ultrasonic_data():
    with ULTRASONIC_INFO_LOCK:
        return dict(GLOBAL_ULTRASONIC_INFO)

def pulse_motors(direction="FWD", duration=MOTOR_PULSE):
    if hasattr(motors, "angle_i_buf"):
        motors.angle_i_buf = 0

    motors.move(direction)
    motors.update_motors()
    time.sleep(duration)

    motors.move("STP")
    motors.update_motors()

# ---------------------------
# X ALIGNMENT
# ---------------------------
def center_on_block_x(name: str, targets: Dict[str, Dict[str, Any]]) -> bool:
    if name not in targets:
        return False

    x = targets[name].get("X")
    if x is None:
        return False

    print(f"[XALIGN] {name}: X = {x:.3f}")

    if abs(x) <= X_TOL:
        print("[XALIGN] Centered")
        return True

    direction = "Right" if x > 0 else "Left"
    print(f"[XALIGN] Turning {direction}")

    motors.turn_and_move(direction, LR_STEP_DURATION)
    time.sleep(PAUSE_WAIT)

    return False

# ---------------------------
# ULTRASONIC APPROACH
# ---------------------------
def approach_and_pickup(block_name):
    while True:
        data = get_ultrasonic_data()
        dist = data.get("US_distance_cm")

        if dist is None:
            time.sleep(0.1)
            continue

        print(f"[ULTRA] {dist:.2f} cm")

        for target_cm, fn in DISTANCE_TO_ACTION_MAP.items():
            if abs(dist - target_cm) <= ULTRA_TOL_CM:
                print(f"[ULTRA] Matched {target_cm} cm")
                fn()
                pulse_motors("REV", 0.1)

                if block_name in unique_blocks:
                    unique_blocks.remove(block_name)

                return

        print("[ULTRA] Forward step")
        pulse_motors("FWD", 0.04)
        time.sleep(0.2)

# ---------------------------
# ARM INIT
# ---------------------------
def init_arm():
    arm.move_to_pose("default")
    time.sleep(0.5)
    arm.move_to_pose("21_84_cm_view")

def shutdown():
    try:
        arm.move_to_pose("default", reverse=True)
    except:
        pass

    try:
        motors.kill_motors()
    except:
        pass

# ---------------------------
# MAIN LOOP
# ---------------------------
def get_closest_block(targets):
    closest = None
    min_d = float("inf")

    for name, info in targets.items():
        D = info.get("D")
        X = info.get("X")
        Y = info.get("Y")
        Z = info.get("Z")
        conf = info.get("confidence", 0)

        print(f"{name}: X={X:.2f} Y={Y:.2f} Z={Z:.2f} D={D:.2f} Conf={conf:.2f}")

        if name not in unique_blocks:
            unique_blocks.append(name)

        if D is not None and D < min_d:
            min_d = D
            closest = name

    return closest

def main():
    print("\n[START] Vision + Ultrasonic Threads")
    vision.start_thread()
    start_ultra_thread(ULTRA_RATE, ULTRA_ECHO, ULTRA_TRIGGER)

    vision.PAUSE_PROCESSING = True
    init_arm()
    vision.PAUSE_PROCESSING = False

    try:
        while True:
            time.sleep(0.1)

            with vision.TARGET_INFO_LOCK:
                targets = dict(vision.GLOBAL_TARGET_INFO)

            if not targets:
                continue

            print("\n======= TARGETS =======")
            target = get_closest_block(targets)

            if target is None:
                continue

            info = targets[target]

            if info.get("D", 999) < VISION_CLOSE_Z and arm.current_pose != "4_21_view":
                print("[ARM] Moving to 4_21_view")
                vision.PAUSE_PROCESSING = True
                arm.move_to_pose("4_21_view")
                vision.PAUSE_PROCESSING = False

            if arm.current_pose == "4_21_view" and info.get("D", 999) < VISION_CLOSE_Z:
                if not center_on_block_x(target, targets):
                    continue

            print(f"[MAIN] Aligned. Approaching {target}")
            vision.PAUSE_PROCESSING = True
            approach_and_pickup(target)
            vision.PAUSE_PROCESSING = False

    except KeyboardInterrupt:
        print("\n[STOPPED]")
    finally:
        print("[CLEANUP]")
        try: vision.stop_thread()
        except: pass
        try: stop_ultra_thread()
        except: pass
        shutdown()
        print("[DONE]")

if __name__ == "__main__":
    main()
