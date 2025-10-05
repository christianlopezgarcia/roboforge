import numpy as np
import time
import board
import busio
import adafruit_bno055

class myIMU:
    
    last_time = 0
    
    
    def __init__(self, i2c=None):
        """Initialize the BNO055 sensor."""
        if i2c is None:
            i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_bno055.BNO055_I2C(i2c)
        self.last_time = time.time()
        self.velocity = [0.0, 0.0, 0.0]  # store estimated speed (x, y, z)

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
        """
        Estimate linear speed by integrating acceleration over time.
        Returns a tuple: (vx, vy, vz) in m/s
        """
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        accel = self.sensor.acceleration  # (ax, ay, az) in m/s^2
        if accel is None:
            return tuple(self.velocity)

        # Integrate acceleration to estimate velocity
        self.velocity = [v + a * dt for v, a in zip(self.velocity, accel)]

        return tuple(self.velocity)
    
    def velocity_to_speed(self):
        self.current_speed = self.velocity[0]^2 + self.velocity[1]^2

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
            
        