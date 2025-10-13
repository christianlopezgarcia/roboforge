import numpy as np
import time
import board
import busio
import adafruit_bno055
import json

class myIMU:
    
    last_time = 0
    accel_cal = (0.0,0.0,0.0)    

    def __init__(self, i2c=None):
        """Initialize the BNO055 sensor."""
        if i2c is None:
            i2c = busio.I2C(board.SCL, board.SDA)
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
        self.current_speed = self.velocity[0]**2 + self.velocity[1]**2

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


        

class myMotors(myIMU):
    
    enable_pid = 1
    
    speed_p = 0
    speed_i = 0
    speed_d = 0
        
    speed_p_limits = [-10,10]
    speed_i_limits = [-10,10]
    speed_d_limits = [-10,10] 
    
    angle_p = 0
    angle_i = 0
    angle_d = 0  
    
    angle_p_limits = [-10,10]
    angle_i_limits = [-10,10]
    angle_d_limits = [-10,10]
    
    speed_total_limits = [speed_p_limits[0]+speed_i_limits[0]+speed_d_limits[0],
                          speed_p_limits[1]+speed_i_limits[1]+speed_d_limits[1]] 
    
    angle_total_limits = [angle_p_limits[0]+angle_i_limits[0]+angle_d_limits[0],
                          angle_p_limits[1]+angle_i_limits[1]+angle_d_limits[1]]
        
    
    def __init__(self,speed_pid,angle_pid):
        super().__init__(i2c=None)
        
        self.speed_p = speed_pid[0]
        self.speed_i = speed_pid[1]
        self.speed_d = speed_pid[2]
        
        self.angle_p = angle_pid[0]
        self.angle_i = angle_pid[1]
        self.angle_d = angle_pid[2] 
    
    def set_desired_angle(self,angle):
        self.desired_angle = angle
        
    def set_desired_speed(self,speed):
        self.desired_speed = speed
    
    #Currently assuming 1000-2000 pwm and    
    def convert_percent_to_output(self,percent):
        pwm = percent * (500/100) + 1000
        return pwm
    
    #Currently waiting on how to output (probably i2c pwm bus)    
    def set_single_motor(self,pin,value):
        x=1
    
    def limit_pid_component(self, value, limits):
        value = max(limits[0], min(value, limits[1]))
        return value
    
    def limit_speed_pid(self):
        self.speed_p_buff = self.limit_pid_component(self.speed_p_buff, self.speed_p_limits)
        self.speed_i_buff = self.limit_pid_component(self.speed_i_buff, self.speed_i_limits)
        self.speed_d_buff = self.limit_pid_component(self.speed_d_buff, self.speed_d_limits)
        
    def limit_angle_pid(self):
        self.angle_p_buff = self.limit_pid_component(self.angle_p_buff, self.angle_p_limits)
        self.angle_i_buff = self.limit_pid_component(self.angle_i_buff, self.angle_i_limits)
        self.angle_d_buff = self.limit_pid_component(self.angle_d_buff, self.angle_d_limits)
        
    def update_speed_pid(self):
        speed_error = self.desired_speed - self.current_speeed
        
        now = time.time()
        dt = now - self.last_time_speed
        
        self.speed_p_buff = speed_error * self.speed_i
        self.speed_i_buff += speed_error*dt * self.speed_i
        self.speed_d_buff =  ((speed_error - self.last_speed_error)/dt)*self.speed_d
        self.limit_speed_pid(self.speed_p_buff,self.speed_i_buff,self.speed_d_buff)
        self.speed_total_buf = self.speed_p_buff + self.speed_i_buff + self.speed_d_buff
        
        self.last_speed_error = speed_error
        self.last_time_speed = time.time()

    def update_angle_pid(self):
        angle_error = self.desired_angle - self.current_speeed
        
        now = time.time()
        dt = now - self.last_time_angle
        
        self.angle_p_buff = angle_error * self.angle_i
        self.angle_i_buff += angle_error*dt * self.angle_i
        self.angle_d_buff =  ((angle_error - self.last_angle_error)/dt)*self.angle_d
        self.limit_angle_pid(self.angle_p_buff,self.angle_i_buff,self.angle_d_buff)
        self.angle_total_buf = self.angle_p_buff + self.angle_i_buff + self.angle_d_buff
        
        self.last_angle_error = angle_error
        self.last_time_angle = time.time()
        

    def update_motor_output_tank(self):
        if(self.speed_total_buf < 0):
            motor_speed_value = self.speed_total_buf*(50/self.speed_total_limits[0])
        elif(self.speed_total_buf > 0):
            motor_speed_value = -self.speed_total_buf*(50/self.speed_total_limits[1])
        else:
            motor_speed_value = 0

        if(self.angle_total_buf < 0):
            motor_angle_value = self.angle_total_buf*(50/self.angle_total_limits[0])
        elif(self.angle_total_buf > 0):
            motor_angle_value = self.angle_total_buf*(50/self.angle_total_limits[1])
        else:
            motor_angle_value = 0       
        
        motor_value_FR = motor_speed_value - motor_angle_value
        motor_value_FL = motor_speed_value + motor_angle_value
        motor_value_BR = motor_speed_value - motor_angle_value
        motor_value_BL = motor_speed_value + motor_angle_value
        
        #set front right motor
        self.set_single_motor(self.pin_FR,self.convert_percent_to_output(motor_value_FR))
        #set front left motor
        self.set_single_motor(self.pin_FL,self.convert_percent_to_output(motor_value_FL))
        #set back right motor
        self.set_single_motor(self.pin_BR,self.convert_percent_to_output(motor_value_BR))
        #set back left motor
        self.set_single_motor(self.pin_BL,self.convert_percent_to_output(motor_value_BL))
    
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
        self.set_single_motor(self.pin_FR,self.convert_percent_to_output(motor_value_FR))
        #set front left motor
        self.set_single_motor(self.pin_FL,self.convert_percent_to_output(motor_value_FL))
        #set back right motor
        self.set_single_motor(self.pin_BR,self.convert_percent_to_output(motor_value_BR))
        #set back left motor
        self.set_single_motor(self.pin_BL,self.convert_percent_to_output(motor_value_BL))
        
            
    def update_motors(self):
        if(self.enable_pid):
           self.update_speed_pid()
           self.update_angle_pid()
           self.update_motor_output_tank()
        else:
            self.update_motor_output_mechanum()

if __name__ == "__main__":
    bno = myIMU()
    i=0
    while True:
        
        
        bno.velocity_to_speed()
        time.sleep(0.1)
        if(i%10 == 0):
            print("Current Angle: " + str(bno.get_angle()[0]) + "deg")
            print("Current Speed: " + str(bno.current_speed) + "m/s")
        i += 1





