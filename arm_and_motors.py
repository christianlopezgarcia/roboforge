import busio
import board
import time

from roboforge_arm_v2.armbackup_1117 import RobotArm
from roboforge_motors.MotorControlv2 import myMotors, BNO055, PCA9685PWM

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
# motors.set_arming_status(1)
# motors.set_angle_pid_enable(1)
# motors.set_desired_speed(0)
# motors.set_desired_angle(motors.current_angle)
motors.set_pid_enable(0)
motors.move("STP")
motors.turn(0)


# i2c = busio.I2C(board.SCL, board.SDA)
# arm = RobotArm(i2c)
# time.sleep(0.5)


# # i2c = busio.I2C(board.SCL, board.SDA)
# bno = BNO055(i2c)
# pca = PCA9685PWM(i2c)

# angle_pid_p = 1.5
# angle_pid_i = 0.1
# angle_pid = [angle_pid_p,angle_pid_i]
# #Create Motors Object
# motors = myMotors(bno,pca,angle_pid)

#Auto Control # arm and initiallize
# motors.set_arming_status(1)
# motors.set_angle_pid_enable(1)
# motors.set_desired_speed(0)
# motors.set_desired_angle(motors.current_angle)

#Update Motors
motors.move("FWD")
motors.update_motors()
time.sleep(.1)
motors.move("STP")
motors.update_motors()
# 
time.sleep(5)
    
print("done")
# motors.kill_motors()  


print('------ DEFAULT --------')
arm.move_to_pose("default")
time.sleep(0.5)
arm.status()

# print('\n------ set_all (hybrid) --------')
# arm.set_all({"base": 105,  "shoulder": 90,  "elbow": 150,"wrist": 0,  "hand": 180}) #cool up right pose WALLE mode
# time.sleep(15)


# print('\n ------Default------')
# arm.move_to_pose("default")
# arm.status()