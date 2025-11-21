import numpy as np
import time
import board
import busio
import adafruit_bno055
import adafruit_pca9685
import json
import math


class BNO055:
    
    last_time = 0
    accel_cal = (0.0,0.0,0.0)    

    def __init__(self, i2c):
        """Initialize the BNO055 sensor."""
        self.sensor = adafruit_bno055.BNO055_I2C(i2c)
        self.last_time = time.time()
        self.velocity = [0.0, 0.0, 0.0]  # store estimated speed (x, y, z)
        self.lin_bias = [0.0, 0.0, 0.0]   # slow-learned bias on linear_acceleration
        self.still_count = 0              # consecutive stationary samples
        self.alpha_bias = 0.02            # bias EMA rate while stationary
        self.vel_leak = 2.0               # 1/s, how fast velocity decays when near-still
        self.v_eps = 0.01                 # m/s: clamp tiny velocities to zero
        self.load_calibration()
        for i in range(1):
            self.calibrate_accel()
        self.last_time = time.time()

    def get_angle(self):
        """
        Get the Euler angles (heading, roll, pitch) in degrees.
        Returns a tuple: (heading, roll, pitch)
        """
        return self.sensor.euler

    def get_angular_speed(self):
        """
        Get angular velocity (gyroscope) in radians/sec.
        Returns a tuple: (x, y, z)
        """
        return self.sensor.gyro

    def get_velocity(self):
        import math

        now = time.time()
        dt = now - self.last_time if self.last_time else 0.0
        # clamp absurd dt (prevents huge jumps after pauses)
        if dt < 0 or dt > 0.5:
            dt = 0.0
        self.last_time = now

        # Prefer linear acceleration (gravity removed)
        accel_raw = self.sensor.linear_acceleration
        if (
            accel_raw is None
            or not isinstance(accel_raw, (tuple, list))
            or len(accel_raw) < 3
            or any(a is None for a in accel_raw)
        ):
            return tuple(self.velocity)

        # subtract slowly-learned bias
        ax = accel_raw[0] - self.lin_bias[0]
        ay = accel_raw[1] - self.lin_bias[1]
        az = accel_raw[2] - self.lin_bias[2]

        # deadband to kill micro-noise (tune 0.05–0.12)
        for i, v in enumerate((ax, ay, az)):
            if abs(v) < 0.08:
                if i == 0: ax = 0.0
                elif i == 1: ay = 0.0
                else: az = 0.0

        # magnitude checks
        accel_mag = math.sqrt(ax*ax + ay*ay)
        gyro = self.sensor.gyro or (0.0, 0.0, 0.0)
        gyro_mag = math.sqrt(gyro[0]**2 + gyro[1]**2 + gyro[2]**2)

        # hysteresis thresholds (tune to your setup)
        ACC_STILL = 0.1   # m/s^2 -> call it "still" if below this
        ACC_MOVE  = 0.15   # m/s^2 -> call it "moving" if above this
        GYR_STILL = 0.02   # rad/s
        GYR_MOVE  = 0.04   # rad/s
        STILL_SAMPLES = 8  # how many consecutive samples to confirm stillness

        # update still counter with hysteresis
        if accel_mag < ACC_STILL and gyro_mag < GYR_STILL:
            self.still_count = min(self.still_count + 1, STILL_SAMPLES)
        elif accel_mag > ACC_MOVE or gyro_mag > GYR_MOVE:
            self.still_count = 0
        # else: inside the hysteresis band; keep current still_count

        if self.still_count >= STILL_SAMPLES:
            # confident we are stationary: hard reset velocity
            self.velocity = [0.0, 0.0, 0.0]
            # trim bias so linear_accel tends to zero at rest
            self.lin_bias[0] = (1 - self.alpha_bias) * self.lin_bias[0] + self.alpha_bias * accel_raw[0]
            self.lin_bias[1] = (1 - self.alpha_bias) * self.lin_bias[1] + self.alpha_bias * accel_raw[1]
            self.lin_bias[2] = (1 - self.alpha_bias) * self.lin_bias[2] + self.alpha_bias * accel_raw[2]
        else:
            # integrate X/Y only
            if dt > 0.0:
                self.velocity[0] += ax * dt
                self.velocity[1] += ay * dt

            # apply velocity leak when nearly still (helps it return to zero fast)
            if accel_mag < ACC_MOVE and dt > 0.0:
                leak = math.exp(-self.vel_leak * dt)  # ~e^{-λ dt}
                self.velocity[0] *= leak
                self.velocity[1] *= leak

        # clamp tiny velocities
        if abs(self.velocity[0]) < self.v_eps: self.velocity[0] = 0.0
        if abs(self.velocity[1]) < self.v_eps: self.velocity[1] = 0.0

        #print(f"Velocity X: {self.velocity[0]:.3f}, Y: {self.velocity[1]:.3f}")
        #print(f"Acceleration X: {ax:.3f}, Y: {ay:.3f}")
        return tuple(self.velocity)
    
    def velocity_to_speed(self):
        self.get_velocity()
        self.current_speed = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        return self.current_speed

    def calibrate_accel(self):
        accel = [0.0, 0.0, 0.0]
        valid_samples = 0

        for i in range(1000):
            accel_num = self.sensor.linear_acceleration

            # Skip if no valid reading or contains None values
            if (
                accel_num is None
                or not isinstance(accel_num, (list, tuple))
                or len(accel_num) < 3
                or any(a is None for a in accel_num)
            ):
                continue

            # Accumulate valid readings
            accel[0] += accel_num[0]
            accel[1] += accel_num[1]
            accel[2] += accel_num[2]
            valid_samples += 1

        # Compute average only if enough valid samples were collected
        if valid_samples > 0:
            self.accel_cal = [
                accel[0] / valid_samples,
                accel[1] / valid_samples,
                accel[2] / valid_samples,
            ]
        else:
            # Fall back to previous calibration if no valid data
            print("Warning: No valid linear_acceleration readings; keeping previous calibration.")

        print(f"Calibration complete. Samples used: {valid_samples}")
        print(f"Calibrated bias: {self.accel_cal}")

    
    CAL_FILE = "calibration.json"

    def save_calibration(self):
        """Save current calibration offsets/radii to a file."""
        cal = self.sensor.calibration_status
        data = {
            "status": tuple(cal),
            "offsets_accelerometer": tuple(self.sensor.offsets_accelerometer),
            "offsets_gyroscope": tuple(self.sensor.offsets_gyroscope),
            "offsets_magnetometer": tuple(self.sensor.offsets_magnetometer),
            "radius_accelerometer": int(self.sensor.radius_accelerometer),
            "radius_magnetometer": int(self.sensor.radius_magnetometer),
        }
        with open(self.CAL_FILE, "w") as f:
            json.dump(data, f)
        print("💾 Calibration saved:", cal)

    def load_calibration(self):
        """Load calibration offsets/radii from file (if present)."""
        try:
            with open(self.CAL_FILE, "r") as f:
                data = json.load(f)

            # The driver handles switching to CONFIG mode internally.
            self.sensor.offsets_accelerometer = tuple(data["offsets_accelerometer"])
            self.sensor.offsets_gyroscope = tuple(data["offsets_gyroscope"])
            self.sensor.offsets_magnetometer = tuple(data["offsets_magnetometer"])
            self.sensor.radius_accelerometer = int(data["radius_accelerometer"])
            self.sensor.radius_magnetometer = int(data["radius_magnetometer"])
            print("✅ Calibration restored from file:", data.get("status"))
            return True
        except FileNotFoundError:
            print("No saved calibration found.")
            return False
        except Exception as e:
            print("⚠️ Failed to load calibration:", e)
            return False
    
    def calibrate_sensor(self):
        x=1

class PCA9685PWM:
    def __init__(self, i2c, freq_hz=1000):
        self.pca = adafruit_pca9685.PCA9685(i2c)
        self.pca.frequency = freq_hz

    def set_raw(self, ch: int, duty16: int):
        """Set raw 16-bit duty (0..65535)."""
        duty16 = max(0, min(65535, int(duty16)))
        self.pca.channels[ch].duty_cycle = duty16

    def set_percent(self, ch: int, percent: float):
        """Set duty by percentage (0..100)."""
        percent = max(0.0, min(100.0, float(percent)))
        duty16 = int(percent / 100.0 * 65535)
        self.set_raw(ch, duty16)

    def all_off(self):
        for ch in range(16):
            self.set_raw(ch, 0)

    def deinit(self):
        self.all_off()
        self.pca.deinit()
        

class myMotors():
    ############### Enable Parameters ###############
    enable_pid_arr = [0,0]
    last_enable_pid_arr = enable_pid_arr
    
    arm_motors = 0
    last_arm_motors = arm_motors
    
    pinAssignments = {"FR":12,"FL":10,"BR":14,"BL":8}
    
    ################## Throttle PID ###############
    speed_p = 0
    speed_i = 0
    
    speed_p_buff = 0
    speed_i_buff = 0
    speed_total_buff = 0
        
    speed_p_limit = 1000
    speed_i_limit = 10
    speed_total_limit = speed_p_limit + speed_i_limit
    ################## Steering PID ###############
    angle_p = 0
    angle_i = 0
    
    angle_p_buff = 0
    angle_i_buff = 0
    angle_total_buff = 0
        
    angle_p_limit = 10
    angle_i_limit = 2
    angle_total_limit = angle_p_limit + angle_i_limit
    
    ################ State Info ##################
    current_speed = 0
    desired_speed = 0
    last_speed_error = 0
    
    current_angle = 0
    desired_angle = 0
    last_angle_error = 0
    
    motor_value_limit = 25 #Percent
    
    def __init__(self,myIMUobj,myPWMobj,speed_pid,angle_pid):
        
        self.myIMU = myIMUobj
        self.myPWM = myPWMobj        
        self.speed_p = speed_pid[0]
        self.speed_i = speed_pid[1]
        
        self.angle_p = angle_pid[0]
        self.angle_i = angle_pid[1]

        self.last_time_speed = time.time()
        self.last_time_angle = time.time()
    
    def set_desired_angle(self,angle):
        self.desired_angle = angle
        
    def set_desired_speed(self,speed):
        self.desired_speed = speed
   
  
    def set_single_motor(self,motor,percent,reverse):
        if(reverse):
            reversePin = self.pinAssignments[motor]
            forwardPin = reversePin + 1
        else:
            forwardPin = self.pinAssignments[motor]
            reversePin = forwardPin + 1
           
        if(percent > 100 or percent < -100):
            print("motor percentage for " + motor + " is out of bounds")
            return 0
        if(percent > 0):
            self.myPWM.set_percent(reversePin,0)
            self.myPWM.set_percent(forwardPin,abs(percent))
        elif(percent < 0):
            self.myPWM.set_percent(forwardPin,0)
            self.myPWM.set_percent(reversePin,abs(percent))
        else:
            self.myPWM.set_percent(forwardPin,0)
            self.myPWM.set_percent(reversePin,0)
    
    def limit_pid_component(self, value, limit):
        value = max(-1*limit, min(value, limit))
        return value
    
    def limit_speed_pid(self):
        self.speed_p_buff = self.limit_pid_component(self.speed_p_buff, self.speed_p_limit)
        self.speed_i_buff = self.limit_pid_component(self.speed_i_buff, self.speed_i_limit)
        
    def limit_angle_pid(self):
        self.angle_p_buff = self.limit_pid_component(self.angle_p_buff, self.angle_p_limit)
        self.angle_i_buff = self.limit_pid_component(self.angle_i_buff, self.angle_i_limit)
        
    def limit_motor_value(self,value):
        if abs(value) < self.motor_value_limit:
            return 0
        else:
            return value
        
    def update_speed_pid(self):
        speed_error = self.desired_speed - self.current_speed
        
        now = time.time()
        dt = now - self.last_time_speed
        
        self.speed_p_buff = speed_error * self.speed_p
        self.speed_i_buff += speed_error*dt * self.speed_i
        self.limit_speed_pid()
        self.speed_total_buff = self.speed_p_buff + self.speed_i_buff
        
        self.last_time_speed = time.time()

    def update_angle_pid(self):
        angle_error = self.desired_angle - self.current_angle
        angle_error_wrapped = self.angle_wrap(angle_error)
        
        now = time.time()
        dt = now - self.last_time_angle

        self.angle_p_buff = angle_error_wrapped * self.angle_p
        self.angle_i_buff += angle_error_wrapped*dt * self.angle_i
        
        self.limit_angle_pid()        

        self.angle_total_buff = self.angle_p_buff + self.angle_i_buff

        self.last_time_angle = time.time()
        

    def update_motor_output_tank(self):
        if(self.speed_total_buff < 0):
            motor_speed_value = self.speed_total_buff*(50/self.speed_total_limit)
        elif(self.speed_total_buff > 0):
            motor_speed_value = -self.speed_total_buff*(50/self.speed_total_limit)
        else:
            motor_speed_value = 0

        if(self.angle_total_buff < 0):
            motor_angle_value = self.angle_total_buff*(50/self.angle_total_limit)
        elif(self.angle_total_buff > 0):
            motor_angle_value = self.angle_total_buff*(50/self.angle_total_limit)
        else:
            motor_angle_value = 0
        
        motor_value_FR = motor_speed_value - motor_angle_value
        motor_value_FL = motor_speed_value + motor_angle_value
        motor_value_BR = motor_speed_value - motor_angle_value
        motor_value_BL = motor_speed_value + motor_angle_value
        
        motor_value_FR = self.limit_motor_value(motor_value_FR)
        motor_value_FL = self.limit_motor_value(motor_value_FL)
        motor_value_BR = self.limit_motor_value(motor_value_BR)
        motor_value_BL = self.limit_motor_value(motor_value_BL)
               
        #set front right motor
        self.set_single_motor("FR", motor_value_FR,1)
        #set front left motor
        self.set_single_motor("FL", motor_value_FL,0)
        #set back right motor
        self.set_single_motor("BR", motor_value_BR,1)
        #set back left motor
        self.set_single_motor("BL", motor_value_BL,0)
    
    def update_motor_output_mechanum(self):
        mask = np.array([ #FR  #FL  #BR  #BL
                        [ 100, 100, 100, 100],#F
                        [   0, 100, 100,   0],#FR
                        [-100, 100, 100,-100],#R
                        [-100,   0,   0,-100],#BR
                        [-100,-100,-100,-100],#B
                        [   0,-100,-100,   0],#BL
                        [ 100,-100,-100, 100],#L
                        [ 100,   0,   0, 100]#FL
                                            ])

        
        section = self.desired_angle/45
        section_f = np.floor(section)+1
        section_low = mask[section_f,:]
        section_high = mask[section_f % 8+1,:]

        motor_value_FR = section_low[1] + (section_high[1] - section_low[1])*section/45
        motor_value_FL = section_low[2] + (section_high[2] - section_low[2])*section/45
        motor_value_BR = section_low[3] + (section_high[3] - section_low[3])*section/45
        motor_value_BL = section_low[4] + (section_high[4] - section_low[4])*section/45
        
        #set front right motor
        self.set_single_motor("FR", motor_value_FR,1)
        #set front left motor
        self.set_single_motor("FL", motor_value_FL,0)
        #set back right motor
        self.set_single_motor("BR", motor_value_BR,1)
        #set back left motor
        self.set_single_motor("BL", motor_value_BL,0)
        
            
    def update_motors(self):
        
        self.current_angle = self.none_check(self.current_angle,self.myIMU.get_angle()[0])
        self.current_speed = self.none_check(self.current_speed,self.myIMU.velocity_to_speed())
        
        if((self.last_enable_pid_arr != self.enable_pid_arr) or (self.last_arm_motors != self.arm_motors)):
            self.reset_motors()
        self.last_arm_motors = self.arm_motors
        self.last_enable_pid_arr = self.enable_pid_arr
                           
        if(self.arm_motors):
            if(self.enable_pid_arr):
                if(self.enable_pid_arr[0]):
                    self.update_speed_pid()
                if(self.enable_pid_arr[1]):
                    self.update_angle_pid()
                self.update_motor_output_tank()
            else:
                self.update_motor_output_mechanum()
        else:
            self.kill_motors()
        


    def kill_motors(self):
        self.set_single_motor("FR",0,1)
        self.set_single_motor("FL",0,0)
        self.set_single_motor("BR",0,1)
        self.set_single_motor("BL",0,0)
        
        self.desired_angle = self.current_angle
        self.desired_speed = 0
        
    def none_check(self,old_data,new_data):
        if(new_data is None):
            print("None Error")
            return old_data
        else:
            return new_data
        
    def angle_wrap(self,angle):
        return (angle + 180.0) % 360.0 - 180.0
    
    def reset_motors(self):
        self.kill_motors()
                
        self.speed_p_buff = 0
        self.speed_i_buff = 0    
        self.angle_p_buff = 0
        self.angle_i_buff = 0
        
        self.current_speed = 0
        self.current_angle = 0
        self.last_speed_error = 0
        self.last_angle_error = 0
    
    def set_speed_pid_enable(self,enable):
        self.enable_pid_arr[0] = enable
        
    def set_angle_pid_enable(self,enable):
        self.enable_pid_arr[1] = enable
        
    def set_both_pid_enable(self,enable):
        self.enable_pid_arr = [enable,enable]
    
    def set_arming_status(self,enable):
        self.arm_motors = enable
        
    def print_state_info(self):
        print("--------------State Info------------------")
        print("Current Speed : " + str(self.current_speed) + "m/s")
        print("Desired Speed : " + str(self.desired_speed) + "m/s")
        print("Current Angle : " + str(self.current_angle) + " Degrees")
        print("Desired Angle : " + str(self.desired_angle) + " Degrees")
        print("")
        
    def print_pid_info(self):
        print("--------------PID Info------------------")
        print("P Buffer Value (Speed): " + str(self.speed_p_buff))
        print("I Buffer Value (Speed)): " + str(self.speed_i_buff))
        print("Total Buffer Value (Speed)): " + str(self.speed_total_buff))
        print("Percent Motor Power (Speed)" + str(self.speed_total_buff/self.speed_total_limit*50)) 
        print("P Buffer Value (Angle): " + str(self.angle_p_buff)) 
        print("I Buffer Value (Angle): " + str(self.angle_i_buff))
        print("Total Buffer Value (Angle): " + str(self.angle_total_buff))
        print("Percent Motor Power (Angle)" + str(self.angle_total_buff/self.angle_total_limit*50))
        print("")

    
if __name__ == "__main__":
    
    #Create I2C Objects
    i2c = busio.I2C(board.SCL, board.SDA)
    bno = BNO055(i2c)
    pca = PCA9685PWM(i2c)
    
    #Create Motors Object
    motors = myMotors(bno,pca,[5000,10],[1.5,0.1])
    
    #Auto Control
    motors.set_arming_status(1)
    motors.set_speed_pid_enable(1)
    motors.set_angle_pid_enable(0)
    
    motors.set_desired_angle(20) #Degrees right +, left - (-180 to 180)
    motors.set_desired_speed(0.1) #m/s
    
    #Direct Control
    #motors.set_single_motor("FR",100,1)
    #motors.set_single_motor("FL",-100,0)
    #motors.set_single_motor("BR",100,1)
    #motors.set_single_motor("BL",-100,0)

    update_time = 0.01 #seconds
    print_time = 1 #seconds
    max_runtime = 20 #seconds
    
    time_start = time.time()
    time_last_update = time.time()
    time_last_print = time.time()
    
    while 1:
        motors.set_desired_speed(0.1)
        #Update Motors
        if(time.time() - time_last_update > update_time):
            motors.update_motors()
            time_last_update = time.time()
            
        #Print Info
        if(time.time() - time_last_print > print_time):
            motors.print_state_info()
            motors.print_pid_info()
            time_last_print = time.time()
            
        #Break out
        rutime_break = ((time.time() - time_start) > max_runtime)   
        if(rutime_break):
            break
        
    print("done")
    motors.kill_motors()  





