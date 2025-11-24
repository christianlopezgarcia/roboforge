# main.py

import time
import busio
import board

from roboforge_arm_v2.armbackup_1117 import RobotArm
from roboforge_motors.MotorControl import myMotors, BNO055, PCA9685PWM
# /home/clopezgarcia2/Desktop/roboforge/roboforge_vision/traingulate_w_yolo.py
from roboforge_vision.traingulate_w_yolo import (
    start_thread,
    stop_thread,
    GLOBAL_TARGET_INFO,
    TARGET_INFO_LOCK
)

i2c = busio.I2C(board.SCL, board.SDA)
arm = RobotArm(i2c)
time.sleep(0.5)



# i2c = busio.I2C(board.SCL, board.SDA)
bno = BNO055(i2c)
pca = PCA9685PWM(i2c)

angle_pid_p = 1.5
angle_pid_i = 0.1
angle_pid = [angle_pid_p,angle_pid_i]
#Create Motors Object
motors = myMotors(bno,pca,angle_pid)

#Auto Control # arm and initiallize
motors.set_arming_status(1)
motors.set_angle_pid_enable(1)
motors.set_desired_speed(0)
motors.set_desired_angle(motors.current_angle)

def run_motors_tmp():
    motors.update_motors()
    motors.set_desired_speed(10)
    motors.update_motors()
    # motors.set_desired_speed(10)
    time.sleep(5)

def teardown():
    print("done")
    motors.kill_motors()

    print('------ DEFAULT --------')
    arm.move_to_pose("default")
    time.sleep(0.5)
    arm.status()


def init_arm():
    print('------ DEFAULT --------')
    arm.move_to_pose("default")
    time.sleep(0.5)
    arm.status()

    print('\n------ set_all (hybrid) --------')
    arm.set_all({"base": 105,  "shoulder": 90,  "elbow": 150,"wrist": 0,  "hand": 180}) #cool up right pose WALLE mode
    time.sleep(15)


def main():
    print("[Main] Starting stereo vision thread...")
    start_thread()

    try:
        init_arm()
        while True:
            time.sleep(0.1)

            # Safely copy the shared target dictionary
            with TARGET_INFO_LOCK:
                targets = dict(GLOBAL_TARGET_INFO)

            if targets:
                print("\n=== TARGETS FROM VISION THREAD ===")
                for name, info in targets.items():
                    print(f"{name}: "
                          f"X={info['X']:.2f} "
                          f"Y={info['Y']:.2f} "
                          f"Z={info['Z']:.2f} "
                          f"D={info['D']:.2f} "
                          f"D={type(info['D'])} "
                          f"D={info['D']} "
                          f"Conf={info['confidence']:.2f}", )
                    # print(info['D'], type(info["D"]))
                    if info['D'] < .24:
                        arm.move_to_pose("focused")
                    # motors.move()

            # YOUR ARM / SERVO / ROBOT DECISIONS GO HERE
            # Example:
            # if 'cube' in targets:
            #     print("Cube found, moving arm...")

    except KeyboardInterrupt:
        print("\n[Main] Keyboard interrupt — shutting down.")


    finally:
        stop_thread()
        time.sleep(0.5)
        print("[Main] Done.")
        teardown()
        time.sleep(5)


if __name__ == "__main__":
    main()
