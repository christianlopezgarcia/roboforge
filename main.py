# main.py

import time
import busio
import board

from roboforge_arm_v2.armbackup_1117 import RobotArm
from roboforge_motors.MotorControl import myMotors, BNO055, PCA9685PWM

# Stereo vision
from roboforge_vision.traingulate_w_yolo import (
    start_thread,
    stop_thread,
    GLOBAL_TARGET_INFO,
    TARGET_INFO_LOCK
)

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

# ----------------------------------------------------
# ARM INIT / TEARDOWN
# ----------------------------------------------------

def teardown():
    print('------ DEFAULT --------')
    arm.move_to_pose("default")
    time.sleep(0.5)
    

    print("done")
    motors.kill_motors()

def init_arm():
    print('------ DEFAULT --------')
    arm.move_to_pose("default")
    time.sleep(0.5)
    
    print('\n------ WIDE VIEW --------')
    arm.move_to_pose('21_84_cm_view')  
    time.sleep(15)

def five_cm_pickup():
    arm.move_to_pose("5cm")
    time.sleep(1)

    arm.move_to_pose("5cm_closed")
    time.sleep(1)
    
    arm.move_to_pose("5cm_30r_closed")
    time.sleep(1)
    
    arm.move_to_pose("default_closed",reverse =True)
    time.sleep(1)
    
    arm.move_to_pose("fold_over",reverse = True)
    time.sleep(1)
    
    arm.move_to_pose("fold_over_open")
    time.sleep(1)
    

def ten_cm_pickup():
    arm.move_to_pose("10_cm")
    time.sleep(1)
    arm.move_to_pose("10_cm_closed")
    time.sleep(1)
    arm.move_to_pose("10_cm_30r_closed")
    time.sleep(1)
    arm.move_to_pose("fold_over",reverse =True)
    time.sleep(1)
    
    arm.move_to_pose("fold_over_open")
    time.sleep(1)
    arm.move_to_pose("default")
    

def eight_cm_pickup():
    arm.move_to_pose("8cm")
    time.sleep(1)
    arm.move_to_pose("8cm_closed")
    time.sleep(1)
    arm.move_to_pose("8cm_30r_closed")
    time.sleep(1)
    arm.move_to_pose("fold_over",reverse =True)
    time.sleep(1)
    
    arm.move_to_pose("fold_over_open")
    time.sleep(1)
    arm.move_to_pose("default")
    

distance_to_move = {5: five_cm_pickup, 8: eight_cm_pickup, 10: ten_cm_pickup}


# ----------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------

def main():
    print("[Main] Starting stereo vision thread...")
    start_thread()

    ultrasonic_running = True   # >>> NEW FLAG <<<

    # Ultrasonic settings
    sample_rate = 100
    echo_pin = 17
    trigger_pin = 27
    start_ultra_thread(sample_rate, echo_pin, trigger_pin)
    try:
        init_arm()
        while True:
            time.sleep(0.1)
            
            # Safely copy the shared target dictionary
            with TARGET_INFO_LOCK:
                targets = dict(GLOBAL_TARGET_INFO)
            # print("\ntargets", targets)
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

                    # Update nearest object
                    if info['D'] < targets[MINIMUM_BLOCK]['D']:
                        MINIMUM_BLOCK = name

                    # Enter focused mode if close
                    if info['D'] < .30 and arm.current_pose != "focused":
                        arm.move_to_pose("focused")
                        time.sleep(0.5)

                # -----------------------------------------------------
                # >>> ULTRASONIC START TRIGGER <<<
                # After focused AND close (< 0.30)
                # -----------------------------------------------------
                if arm.current_pose == "focused" and targets[MINIMUM_BLOCK]['D'] < .30:
                    D_min = targets[MINIMUM_BLOCK]['D']
                    print(f"Closest distance: {D_min:.3f}")


                # -----------------------------------------------------
                # >>> READ ULTRASONIC (if running) <<<
                # -----------------------------------------------------
                with ULTRASONIC_INFO_LOCK:
                    uinfo = dict(GLOBAL_ULTRASONIC_INFO)

                for name, info in uinfo.items():
                    print(f"ULTRA {name}: "
                            f"{info['US_distance_cm']:.2f} cm "
                            f"(ts={info['ts']:.2f})")

    except KeyboardInterrupt:
        print("\n[Main] Keyboard interrupt — shutting down.")

    finally:
        stop_thread()
        if ultrasonic_running:
            stop_ultra_thread()

        time.sleep(0.5)
        teardown()
        time.sleep(1)
        print("[Main] Done.")


if __name__ == "__main__":
    main()
