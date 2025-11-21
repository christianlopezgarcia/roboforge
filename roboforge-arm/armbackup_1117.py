#Class Structure for robot arm
import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from adafruit_servokit import ServoKit
## Join class for DH table to determine current position
import numpy as np

class joint:
    
    def __init__(self, theta, d, alpha, a):
        self.theta = theta
        self.alpha = alpha
        self.a = a
        self.d = d
    
    def A(self):
        return A_i(self.theta, self.alpha, self.a, self.d)

    def add_angle(self, theta = 0):
        self.theta = self.theta + theta

## ----------------------------------------------------------------------
# Basic functions for xarm

# Compute the A matrix for a single joint
def A_i(th_i, alpha_i, a_i, d_i):

    # np uses radians 
    th_i = np.deg2rad(th_i)
    alpha_i = np.deg2rad(alpha_i)

    #Create A matrix for 
    A = np.array([[np.cos(th_i), -np.sin(th_i)*np.cos(alpha_i), np.sin(th_i)*np.sin(alpha_i), a_i*np.cos(th_i)],
              [np.sin(th_i), np.cos(th_i)*np.cos(alpha_i), -np.cos(th_i)*np.sin(alpha_i), a_i*np.sin(th_i)], 
              [0, np.sin(alpha_i), np.cos(alpha_i), d_i],
              [0,0,0,1]])
    return A

# Multiply 6 joints Transform matricies together for the forward kinematics -- gives the matrix defining the 
#   Transformation from the end effector to the base frame -- should be in cartesian coordinates?
def multiply_all(joints):
    length = len(joints)
    # use np matmul to multiply all matricies one by one A1 * A2 * A3 * A4 * A5 * A6
    T = joints[0].A()
    for i in range(1,length):
        T = np.matmul(T,joints[i].A())
    
    
    return T

# grabs the last column of a matrix for extracting the cartesian coordinates, assuming the end effector is 
#   the final coordinate frame
#       Otherwise an offset computation is needed
def get_carte(A):
    cart = A[0:3,3]
    return cart


class robo_arm:
    
    
    def __init__(self):
        # red cube vs blue cube vs green cube?
        # boolean for each? so we only pick up one of each. 
        # -----------------------------
        # Joint setup -- angles at Zero -- true angles. not full 180 degree range on servos -- means there is some mapping...
        # -----------------------------
        # this performs the lateral move to under the base servo
        self.ping2base_j = joint(180, 0, 0, 45)

        self.base2shoulder_j = joint(-90  , 0 ,  0 , 70) # rotate back up - forward - and extend to the shoulder servo pivot
        # from shoulder to the elbow  -- rotate back or CCW -> positive -- then out 65mm to the elbow pivot
        #   --shoulder servo angled backward to angle zero -- not quite 90, maybe 85?
        self.shoulder2elbow_j = joint(90, 0   , 0 , 65)
        # from the elbow to the wrist
        #   --elbow servo angle set to zero -- measured 57 degrees CCW
        self.elbow2wrist_j = joint(57, 0   , 0 , 65)
        #from wrist to grabber
        #   --wrist angle set to zero, measured 64 degrees CW
        self.wrist2effector_j = joint(-64, 0, 0  , 90)

        
        # -----------------------------
        # PCA9685 setup
        # -----------------------------
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c)
        self.pca.frequency = 50  # standard servo frequency
        # kit = ServoKit(channels=16)  # For standard small servos

        large_servos = [7]          # high-torque / metal gear
        small_servos = [13, 14, 15] # standard hobby/RC servos

        self.kit = ServoKit(channels=16)
        super().__init__() # something along these lines

       
      
        # Stops the update --not sure if this will work -- 11/17 unused
        self.stop = False 

        self.servo_angles_default = {
            "base" : 90,
            "shoulder" : 180, # servo 5
            "elbow" : 0,  # servo
            "wrist" : 0,   # servo 
            "hand" : 180
        }

        # looks down towards ground at 45 degrees
        self.servo_angles_45down = {
            "base" : 0,  # -- check this
            "shoulder" : 90, # servo 5 -- check this  maybe slightly back from 90
            "elbow" : 180,  # servo
            "wrist" : 0,   # servo 
            "hand" : 180
        }
        
        self.servo_angles_safe = {
            "base" : 0,  # -- check this
            "shoulder" : 180, # servo 5 -- check this  maybe slightly back from 90
            "elbow" : 0,  # servo
            "wrist" : 180,   # servo 
            "hand" : 180
        }

        ## -------- Angles mostly for Calibration ------------------
        self.servo_angles_straight_up = {
            "base" : 0, 
            "shoulder" : 90, 
            "elbow" : 90,  # servo
            "wrist" : 90,   # servo 
            "hand" : 90
        }

        self.servo_angles_reach_forward = {
            "base" : 180, 
            "shoulder" : 180, # servo 5
            "elbow" : 85,  # servo
            "wrist" : 90,   # servo 
            "hand" : 180
        }

        self.servo_angles_elbow_l = {
            "base" : 0, 
            "shoulder" : 90, 
            "elbow" : 180,  # servo
            "wrist" : 90,   # servo 
            "hand" : 180
        }
        self.servo_angles_wrist_l = {
            "base" : 0, 
            "shoulder" : 90, 
            "elbow" : 90,  # servo
            "wrist" : 0,   # servo 
            "hand" : 180
        }
        ## ^^^^^^^^^^^^^^^ Angles mostly for Calibratio ^^^^^^^^^^^


        #set current angles to default
        self.current_servo_angles = self.servo_angles_default

        self.next_servo_angles = self.servo_angles_default
        #set default angles
        # self.servo_angles = {
        #     "base" : 90,
        #     "shoulder" : 0,
        #     "elbow" : 135,
        #     "wrist" : 90,
        #     "hand" : 180
        # }

        # self.servo_list = [wrist]
        self.servo_channel = {
        "base" : 6,
        "shoulder" : 5, # Servo 5
        "elbow" : 4,    # Servo 4
        "wrist" : 3,    # Servo 3
        "hand" : 1      # Servo 1
        }

        self.large_servo_objects = {}
        print("\n ---- INIT -----")
        for servo_joint in self.servo_angles_default:

            try:
                
                ch = self.servo_channel[servo_joint]
                
                if(servo_joint == "shoulder"):
                    #we are a t big servo -- do it difff
                    s = servo.Servo(self.pca.channels[ch], min_pulse=500, max_pulse=2500)
                    self.large_servo_objects[ch] = s
                    s.angle = self.servo_angles_default["shoulder"]
                    print(f"servo_joint: {servo_joint}, ch: {ch}, angle: {s.angle}")

                    time.sleep(0.05)
                    time.sleep(1)
                else:
                    # print(ch)
                    self.kit.servo[ch].angle = self.servo_angles_default[servo_joint] 
                    # Debugging
                    # print("set channel to :")
                    # print(self.kit.servo[ch].angle)

                    print(f"servo_joint: {servo_joint}, ch: {ch}, angle: {self.kit.servo[ch].angle}")
            except Exception:
                print("failed to update channel: "+ ch)
                pass

        # servo0 = adafruit_motor.servo.Servo(self.servo_channel["elbow"])
        print("\n\n")    
    # END OF INIT

    def teardown(self):
        """Instead of stopping the PCA9685, save duty cycles."""
        self.saved_duty = []

        for i in range(16):
            try:
                dc = self.pca.channels[i].duty_cycle
                self.saved_duty.append(dc)
                self.pca.channels[i].duty_cycle = 0  # stop servo motion
            except:
                self.saved_duty.append(None)



    def move_to(self, servo_obj, current, target, step=2, delay=0.02):
        """
        Gradually move a large servo to the target angle
        """
        target = max(0, min(180, int(round(target))))
        if target == current:
            return current

        step = max(1, int(abs(step)))

        if target > current:
            angle = current
            while angle < target:
                angle = min(angle + step, target)
                servo_obj.angle = angle
                time.sleep(delay)
        else:
            angle = current
            while angle > target:
                angle = max(angle - step, target)
                servo_obj.angle = angle
                time.sleep(delay)

        return target  # updated current angle 
    # # define new servo angle -- by keyword atm 
    # def set_angle(self, **servos):    
        
    #     for servo, angle in servos.items():
    
    #         try:
    #             ch = self.servo_channel[servo]
    #             print(ch)
    #             #Save the angle in object dict
    #             self.servo_angles[servo] = angle
    #             #Update that servo to be the angle
    #             kit.servo[ch].angle = self.servo_angles[angle]
    #         except Exception:
    #             pass
    
    # simple update for the angles we changed
    def update(self, delay = 0.5):
        #optional delay
        time.sleep(delay)

        #iterate through servos and update the channel
        for servo_joint in self.current_servo_angles:
            ch = self.servo_channel[servo_joint]
            # print("update channgel: ")
            # print(ch)
            # try:
            if(self.stop == False):
                    
                    #Mr christian code rapper // or "elbow"
                
                if(servo_joint == "shoulder"):
                    # TODO fix reduncadncy
                    s = self.large_servo_objects[ch]
                    current_angle = self.current_servo_angles[servo_joint]  
                    next_angle = self.next_servo_angles[servo_joint]

                    current_angle = self.move_to(s, current_angle, next_angle)
                    self.current_servo_angles[servo_joint] = current_angle
                    print(f"servo_joint: {servo_joint}, ch: {ch}, angle: {s.angle}")
                    
                    # time.sleep(delay) # a bit extra in there for large servo
                #Standard servos
                else:
                    ch = self.servo_channel[servo_joint]
                    self.kit.servo[ch].angle = self.next_servo_angles[servo_joint]  
                    self.current_servo_angles[servo_joint] = self.next_servo_angles[servo_joint]
                    # Debugging
                    # print("set channel to :")
                    # print(ch)
                    print(f"servo_joint: {servo_joint}, ch: {ch}, angle: {self.kit.servo[ch].angle}")
            else:
                break
            
            # except Exception:
                print("exception thrown in update")
                print(servo_joint)
                pass
        print("Robo-Arm Angle Update Complete")

    # actual values from servo channel
    def get_servo_angles(self):

        return [self.kit.servo[self.servo_channel["base"]].angle, 
                self.large_servo_objects[7].angle, 
                self.kit.servo[self.servo_channel["elbow"]].angle,
                self.kit.servo[self.servo_channel["wrist"]].angle,
                self.kit.servo[self.servo_channel["hand"]].angle]

    def get_x_y(self):
        ''' 
            Use the DH table joints and Add angle -- always theta -- to compute the current x/y coordinatae for the end effector --
                -- rather, where the end effector will CLOSE

            -   some calibration may be necessary
            -   


            shoulder -- 90 is straight up: Add 0 degrees to theta. 180 is forward: add -90 to theta. 0 is back: +90 to theta
                        servo: 0   -> joint 

                        servo maps + 90 degrees in theta, and extends ~65mm
            elbow    -- 
        j
        
        '''
        print("shoulder angle")
        print(self.current_servo_angles["shoulder"])
        # subtract because negative rotation is forward, 180 is full forward for the soulder
        self.shoulder2elbow_j.add_angle(-self.current_servo_angles["shoulder"]) # approximate with what we've got. this one's range is decent

        self.elbow2wrist_j.add_angle(-self.kit.servo[self.servo_channel["elbow"]].angle) # 
        self.wrist2effector_j.add_angle(self.kit.servo[self.servo_channel["wrist"]].angle)
        
        all_joints = [self.ping2base_j, self.base2shoulder_j, self.shoulder2elbow_j, self.elbow2wrist_j, self.wrist2effector_j]
        T06 = multiply_all(all_joints)
        xyz = get_carte(T06)
        return xyz
        
        # return self.servo_angles
        #easy pose /update -- just get the order right
    def pose(self, 
             base = 90,
             shoulder = 180,
             elbow = 0,
             wrist = 0,
             hand = 180):
            

        
        #base is 7V and not used atm
        self.next_servo_angles["base"] = base
        self.next_servo_angles["shoulder"] = shoulder
        self.next_servo_angles["elbow"] = elbow
        self.next_servo_angles["wrist"] = wrist
        self.next_servo_angles["hand"] = hand

        # self.servo_angles["hand"] = hand
        self.update()

    #return to netrual -- update included
    def pose_neutral(self):
        self.next_servo_angles["shoulder"] = 180
        self.next_servo_angles["elbow"] = 135
        self.next_servo_angles["wrist"] = 45
        # self.next_servo_angles["hand"]  = 180
        self.update()

    '''
    #currently deprecated from the servo update 11/17
    def pose_ramp(self,shoulder, elbow, wrist):
        step_deg = 1
        step_sec = 0.02
        #determine the direction of travel for each limb

        # negative if set angle is less than current angle
        # shoulder_dir = self.servo_angles["shoulder"] - shoulder
        elbow_diff = elbow - self.servo_angles["elbow"]  
        wrist_diff = wrist - self.servo_angles["wrist"]
        
        while(
            # self.servo_angles["shoulder"] != shoulder + step
            # self.servo_angles["elbow"] < elbow
            abs(elbow_diff) - step_deg > 0 or abs(wrist_diff) - step_deg > 0
        ):
            if(elbow_diff > 0):
                #keep this 
                self.servo_angles["elbow"] = self.servo_angles["elbow"] + step_deg
                #Try directly setting to ease some jitter -- still does the update
                self.kit.servo[self.servo_channel["elbow"]].angle = self.servo_angles["elbow"] + step_deg
            elif(elbow_diff < 0):
                self.servo_angles["elbow"] = self.servo_angles["elbow"] - step_deg
                self.kit.servo[self.servo_channel["elbow"]].angle = self.servo_angles["elbow"] - step_deg

            if(wrist_diff > 0):
                self.servo_angles["wrist"] = self.servo_angles["wrist"] + step_deg
                self.kit.servo[self.servo_channel["wrist"]].angle = self.servo_angles["wrist"] + step_deg
            elif(wrist_diff < 0):
                self.servo_angles["wrist"] = self.servo_angles["wrist"] - step_deg
                self.kit.servo[self.servo_channel["wrist"]].angle = self.servo_angles["wrist"] - step_deg
            
            # don't update if we do it here manually
            # self.update(0)
            elbow_diff = elbow - self.servo_angles["elbow"]
            wrist_diff = wrist - self.servo_angles["wrist"]
            # time.sleep(step_sec)
    '''
    # Sets the robot to look down towards block
    def pose_45down(self):
        self.next_servo_angles = self.servo_angles_45down
        self.update()

    # def rotate(self, base = 0)
    
    #Set the grib strength, barely touching is default. 180 is wide open
    def grab(self, strength = 120):
        self.next_servo_angles["hand"] = strength
        self.update(0.5)
        time.sleep(0.5)

    #release grip to 100% open
    def release(self):
        self.servo_angles["hand"] = 180
        self.update()
    

    # always close out here -- won't drop!
    def pose_go_home(self):
        self.next_servo_angles = self.servo_angles_default
        # self.next_servo_angles["hand"] = grip # default grab 120 atm
        self.update()
        
    # always close out here -- won't drop!
    # 90, 120, 0
    def pose_stand_up_look_forward(self, grip = 120):
        self.next_servo_angles = self.servo_angles_default
        self.next_servo_angles["hand"] = grip # default grab 120 atm
        self.update()

    # for comparing the DH table
    def pose_reach_forward(self):
        self.next_servo_angles = self.servo_angles_reach_forward
        self.update()

    # for comparing the DH table
    def pose_straight_up(self):
        self.next_servo_angles = self.servo_angles_straight_up
        self.update()

    def pose_elbow_L(self):
        self.next_servo_angles = self.servo_angles_elbow_l
        self.update()

    def pose_wrist_L(self):
        self.next_servo_angles = self.servo_angles_wrist_l
        self.update()

    # def close_down(self):
    #     self.pose_go_home()
    #     print("\nCleaning up PCA9685")
    #     self.pca.deinit()
        
# rotate the base, rotate the wrist up
# look right, 
# if somethin's found: returns the angle of the base with the arm pointed at the detected object
# if nothing's found: returns 0
# ** Untested 11-17
def look_left(self):
    self.servo_angles["wrist"] = 45 # guessing
    self.cube_found = False

    step_size = 1 #? 2?
    # need a ramp function for the base -- and a stop function

    # while object not detected
    # rotate base left
    while(self.servo_angles["base"] + step_size < 180 and not cube_found):
        self.servo_angles["base"] = self.servo_angles["base"] + step_size

    if(cube_found):
        return self.kit.servo[self.servo_channel["base"]].angle
    else:
        # look back
        ''' or wherever midpoint is'''
        while(self.servo_angles["base"] - step_size > 90 and not cube_found):
            self.servo_angles["base"] = self.servo_angles["base"] + step_size

        if(cube_found):
            # small chance we catch it on the way back
            return self.kit.servo[self.servo_channel["base"]].angle
        else:
            # we're back at neutral. -- nothing found
            return 0
    self.next_servo_angles = self.servo_angles_default

# Guessing on the polarity of these
# Lifts the wrist up to look horizontally, looks right, 
# if somethin's found: returns the angle of the base with the arm pointed at the detected object
# if nothing's found: returns 0
# untested 11-17
def look_right(self):
    self.servo_angles["wrist"] = 45 # guessing
    self.cube_found = False

    step_size = 1 #? 2?
    # need a ramp function for the base -- and a stop function

    # while object not detected
    # rotate base left

    # Use this same logic to prevent overflow on the robo- ramp function
    # this will have to change for the 
    while(self.servo_angles["base"] - step_size > 0 and not cube_found):
        self.servo_angles["base"] = self.servo_angles["base"] - step_size

    if(self.cube_found):
        return self.kit.servo[self.servo_channel["base"]].angle
    else:
        # look back
        ''' or wherever midpoint is'''
        while(self.servo_angles["base"] + step_size < 90 and not self.cube_found):
            self.servo_angles["base"] = self.servo_angles["base"] + step_size

        if(self.cube_found):
            # small chance we catch it on the way back
            return self.kit.servo[self.servo_channel["base"]].angle
        else:
            # we're back at neutral. 
            return 0

# we need a way to continously track the object? 
# def track(self):
    # reference gets posted - a distance or an angle 
    # we minimize the difference between the reference and our value
if __name__ == "__main__":
    arm = robo_arm()
    time.sleep(1)

    arm.pose(
        # base = 90,
        # shoulder = 90,
        elbow = 100,
        wrist = 40,
        hand = 180
        )
    time.sleep(1)
    arm.pose(
        # base = 90,
        # shoulder = 90,
        elbow = 100,
        wrist = 40,
        hand = 0
        )
    time.sleep(1)
    arm.pose(
        # base = 90,
        # shoulder = 90,
        # elbow = 0,
        # wrist = 40,
        hand = 90
        )
    # arm.teardown()

   
    # arm.pose_elbow_L()
    # arm.get_x_y()
    # time.sleep(1)

    # arm.pose_straight_up()
    # arm.get_x_y()
    # time.sleep(2)
    # time.sleep(2)

    # arm.pose_reach_forward()
    # arm.get_x_y()
    # time.sleep(2)



    # arm.pose_go_home()
    print("This code runs when my_module.py is executed directly.")
