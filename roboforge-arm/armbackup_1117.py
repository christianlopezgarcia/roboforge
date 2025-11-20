

!pip install adafruit-circuitpython-servokit
#Class Structure for robot arm

import time
# from adafruit_servokit import ServoKit

class robo_arm:
    
    
    def __init__(self, sim = False):
        
        # Flag to potentially stop the arm during a movement -- another thread would have to access it. 
        self.stop = False

        if(sim == False):
            self.kit = ServoKit(channels=16)
            super().__init__() # something along these lines
        
            
        self.servo_angles_default = {
            "base" : 90, 
            "shoulder" : 0, # servo 5
            "elbow" : 135,  # servo
            "wrist" : 90,   # servo 
            "hand" : 180
        }

        # Not feasible with bot :/
        # self.servo_angles_45stand = {
        #     "base" : 0,  # -- check this
        #     "shoulder" : 90, # servo 5 -- check this  maybe slightly back from 90
        #     "elbow" : 180,  # servo
        #     "wrist" : 0,   # servo 
        #     "hand" : 180
        # }

        # looks down towards ground at 45 degrees
        self.servo_angles_45down = {
            "base" : 0,  # -- check this
            "shoulder" : 90, # servo 5 -- check this  maybe slightly back from 90
            "elbow" : 180,  # servo
            "wrist" : 0,   # servo 
            "hand" : 180
        }


        #set current angles to default
        self.servo_angles = self.servo_angles_default
        # {
        #     "base" : 90,
        #     "shoulder" : 0,
        #     "elbow" : 135,
        #     "wrist" : 90,
        #     "hand" : 180
        # }

        self.servo_channel = {
        "base" : 0,
        "shoulder" : 12,
        "elbow" : 15,
        "wrist" : 14,
        "hand" : 13
        }

        for servo in self.servo_angles:
            try:
                ch = self.servo_channel[servo]
                # print(ch)
                if(sim == False):
                    self.kit.servo[ch].angle = self.servo_angles[servo] 
            except Exception:
                print("failed to update channel: "+ ch)
                pass

        print("ROBO-ARM INIT COMPLETE")    
    # END OF INIT
     
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
    def update(self, delay = 1):
        #optional delay
        time.sleep(delay)
        if(self.sim == False):
            #iterate through servos and update the channel on the breakout board
            for servo in self.servo_angles:
                print("update channgel: ")
                print(ch)
                try:
                    if(self.stop == False):
                        ch = self.servo_channel[servo]
                        self.kit.servo[ch].angle = self.servo_angles[servo]  
                    else:
                        break
                except Exception:
                    pass
        else:
            print(self.servo_angles)
        print("Robo-Arm Angle Update Complete")


    def get_servo_angles(self):

        return [self.kit.servo[self.servo_channel["base"]].angle, 
                self.kit.servo[self.servo_channel["shoulder"]].angle, 
                self.kit.servo[self.servo_channel["elbow"]].angle,
                self.kit.servo[self.servo_channel["wrist"]].angle,
                self.kit.servo[self.servo_channel["hand"]].angle]
    

        #easy pose /update -- just get the order right
    def pose(self, shoulder = 90, elbow = 90, wrist = 45):
        
        #shoulder is 7V and not used atm
        self.servo_angles["shoulder"] = shoulder
        self.servo_angles["elbow"] = elbow
        self.servo_angles["wrist"] = wrist
        # self.servo_angles["hand"] = hand
        self.update()

    #return to netrual -- update included -- find macros for the deg
    def pose_neutral(self):
        self.servo_angles = self.servo_angles_default
        # self.servo_angles["shoulder"] = 180
        # self.servo_angles["elbow"] = 135
        # self.servo_angles["wrist"] = 45
        # self.servo_angles["hand"]  = 180
        self.update()


    # Ramp not working when set to same angle edges?
    def pose_ramp(self,shoulder, elbow, wrist):
        step_deg = 1.0
        step_sec = 0.005
        #determine the direction of travel for each limb

        # negative if set angle is less than current angle
        # shoulder_dir = self.servo_angles["shoulder"] - shoulder
        elbow_diff = elbow - self.servo_angles["elbow"]  
        wrist_diff = wrist - self.servo_angles["wrist"]
        
        while(
            # self.servo_angles["shoulder"] != shoulder + step
            # self.servo_angles["elbow"] < elbow
            # an Exit flag may be helpful
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

    # Sets the robot to look down towards block
    def pose_45down(self):
        self.servo_angles = self.servo_angles_45down
        self.update()


    #Set the grib strength, barely touching is default. 180 is wide open
    def grab(self, strength = 120):
        self.servo_angles["hand"] = strength
        self.update(0.5)
        time.sleep(0.5)

    #release grip to 100% open
    def release(self):
        self.servo_angles["hand"] = 180
        self.update()
    
    #Stop command
    
    #gradual command 
    # give a joint and direction, slowly move that joint until a flag is set, or maximum is reached

    # def slow

    #gentle adjust command
    # for fine adjustments -- 3 degrees, 1 degree, etc. add this to the current pose angles and update
    
    # for looking around
    def look_down(self, degrees):
        self.servo_angles["wrist"] = self.servo_angles["wrist"] + degrees
        self.update()
     


    

# Determine the servo ranges
arm = robo_arm()
arm.pose(0,0,0)


## DH Table for the Roboforge Arm - Forwrad Kinematics
# Allows us to determine the x-z position of the robot. 
## Inverse Kinematics for Roboforge manipulator -- Assumes base is stationary -- 2x2 
#Simple test for Arm HIL
arm = robo_arm()

print("Robo-Arm's Servo Channels used on breakout board")
print(arm.servo_channel)

print("Robo-Arm's Initial servo angles ")
print(arm.servo_angles)


#Simple pose -- snap right to angles
#angles are shoulder, elbow, wrist -- use grab to change hand angle
arm.pose(0, 0, 180)

arm.grab()

arm.pose_ramp(0,45,120)

arm.release()

# Sim for the arm -- print out the angles
# import robo_arm
arm = robo_arm(sim = True)
# print(arm.servo_angles)
arm.pose(0,0,0)
# print(arm.servo_angles)
