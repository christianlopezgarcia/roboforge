# main.py

import time
import busio
import board

from roboforge_arm_v2.armbackup_1117 import RobotArm
from roboforge_motors.MotorControlv2 import myMotors, BNO055, PCA9685PWM

# Stereo vision (updated import)
import roboforge_vision.traingulate_w_yolo as vision

# >>> ULTRASONIC ADD >>>
from Ultrasonic import (
    start_thread as start_ultra_thread,
    stop_thread as stop_ultra_thread,
    GLOBAL_ULTRASONIC_INFO,
    ULTRASONIC_INFO_LOCK
)
# <<< ULTRASONIC END <<<

# ----------------------------------------------------
# INIT HARDWARE
# ----------------------------------------------------

i2c = busio.I2C(board.SCL, board.SDA)
arm = RobotArm(i2c)
time.sleep(0.5)

bno = BNO055(i2c)
pca = PCA9685PWM(i2c)

angle_pid_p = 1.5
angle_pid_i = 0.1
angle_pid = [angle_pid_p, angle_pid_i]

# Create Motors Object
motors = myMotors(bno, pca, angle_pid)

# Auto Control # arm and initialize
motors.set_pid_enable(0)
motors.move("STP")
motors.turn(0)

def run_motors_tmp():
    motors.update_motors()
    motors.set_desired_speed(10)
    motors.update_motors()
    time.sleep(5)

# ----------------------------------------------------
# ARM INIT / TEARDOWN
# ----------------------------------------------------

def teardown():
    print('------ DEFAULT --------')
    arm.move_to_pose("default", reverse=True)
    time.sleep(0.5)
    motors.kill_motors()
    print("done")

def init_arm():
    print('------ DEFAULT --------')
    arm.move_to_pose("default")
    time.sleep(0.5)

    print('\n------ WIDE VIEW --------')
    arm.move_to_pose('21_84_cm_view')
    # time.sleep(15)

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
    arm.move_to_pose("4_21_view",reverse = True)


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
    arm.move_to_pose("4_21_view",reverse = True)

    # time.sleep(1)
    # arm.move_to_pose("default")

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
    arm.move_to_pose("4_21_view",reverse = True)
    # time.sleep(1)
    # arm.move_to_pose("default")

DISTANCE_TO_ACTION_MAP = {5: five_cm_pickup, 8: eight_cm_pickup, 10: ten_cm_pickup}

def get_ultrasonic_data():
    with ULTRASONIC_INFO_LOCK:
        uinfo = dict(GLOBAL_ULTRASONIC_INFO)
    return uinfo

def move_1ms_motors(direction = "FWD"):
    print("StART")
    motors.move(direction)
    motors.update_motors()
    time.sleep(0.1)
    print("STOP")
    motors.move("STP")
    motors.update_motors()
    motors.kill_motors()

def approach_and_pickup():
    TOL = 0.5
    i = 1

    while True:
        print("loop, i:", i)
        data = get_ultrasonic_data()
        current_distance = data.get('US_distance_cm')

        if current_distance is None:
            print("No ultrasonic reading. Waiting...")
            time.sleep(0.1)
            continue

        print("Distance:", current_distance)

        matched = False
        for target_dist, pickup_fn in DISTANCE_TO_ACTION_MAP.items():
            if abs(current_distance - target_dist) <= TOL:
                print(f"Distance {current_distance} within ±{TOL} of {target_dist}. Executing pickup.")
                pickup_fn()
                matched = True
                move_1ms_motors(direction = "REV")
                break
        if matched:
            break

        if not matched:
            move_1ms_motors()
        time.sleep(.5)
        i += 1

# ----------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------
unique_blocks = []

def main():
    print("[Main] Starting stereo vision thread...")
    vision.start_thread()

    ultrasonic_running = True

    sample_rate = 100
    echo_pin = 17
    trigger_pin = 27
    start_ultra_thread(sample_rate, echo_pin, trigger_pin)

    print("ENTER WHILE LOOP")
    try:
        vision.PAUSE_PROCESSING = True
        init_arm()
        vision.PAUSE_PROCESSING = False
        while True:
            time.sleep(0.1)

            with vision.TARGET_INFO_LOCK:
                targets = dict(vision.GLOBAL_TARGET_INFO)

            if targets:
                print("\n=== TARGETS FROM VISION THREAD ===")
                MINIMUM_BLOCK = list(targets.keys())[0]
                for name, info in targets.items():
                    print(f"{name}: "
                          f"X={info['X']:.2f} "
                          f"Y={info['Y']:.2f} "
                          f"Z={info['Z']:.2f} "
                          f"D={info['D']:.2f} "
                          f"Conf={info['confidence']:.2f}")

                    if name not in unique_blocks:
                        unique_blocks.append(name)

                    if info['D'] < targets[MINIMUM_BLOCK]['D']:
                        MINIMUM_BLOCK = name

                if info['D'] < 0.30 and arm.current_pose != "4_21_view":
                    vision.PAUSE_PROCESSING = True
                    arm.move_to_pose("4_21_view",)
                    vision.PAUSE_PROCESSING = False
                    time.sleep(0.5)

                if arm.current_pose == "4_21_view" and targets[MINIMUM_BLOCK]['D'] < 0.30:
                    D_min = targets[MINIMUM_BLOCK]['D']
                    print(f"Closest distance: {D_min:.3f}")

                print("APPROACHING AND PICKING UP")
                vision.PAUSE_PROCESSING = True
                time.sleep(3)
                approach_and_pickup()
                vision.PAUSE_PROCESSING = False

    except KeyboardInterrupt:
        print("\n[Main] Keyboard interrupt — shutting down.")

    finally:
        vision.stop_thread()
        if ultrasonic_running:
            stop_ultra_thread()

        time.sleep(0.5)
        teardown()
        time.sleep(1)
        print("[Main] Done.")

if __name__ == "__main__":
    main()
