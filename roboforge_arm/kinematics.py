# Forward and Inverse Kinematics for the robo arm
import numpy as np
import math

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

# def base_frame([x,y]):
#     # simple add/subtract to convert forward frame of shoulder @ 180 to robot frame
#     # robot frame based on the lower Ultrasonic sensor

def frame_rot(degrees, vector):
    th = np.deg2rad(degrees)

   # put it in the Q1 reference frame:
    rot = np.array([[np.cos(th),np.sin(th),0],
                   [-np.sin(th),np.cos(th),0],
                   [0,          0,         1]])
    # print(rot @ vector)
    return rot @ vector

def kin_fwd(th1, th2):

    # perform same rotations? ... yes rigjt?

    # measured lengths
    length1 = 65 # mm
    length2 = 65 # mm
    length_hand = 90 # mm

    # limit the angles
    limits = [0,180] 
    th1 = max(limits[0], min(limits[1], th1))
    th2 = max(limits[0], min(limits[1], th2))

    # convert to radians for np
    th1 = np.deg2rad(th1)
    th2 = np.deg2rad(th2)

    # will return simply the xy coordinates, given the current angles
    x = length1*np.cos(th1) + length2*np.cos(th1 + th2)
    y = length1*np.sin(th1) + length2*np.sin(th1 + th2)

    xy = [x,y]
    # xy = base_frame(xy) # TODO
    return xy

# input current position and desired position 
# -> returns theta1 theta2 for the 2 link robot manipulator
def kin_inv(x_des, y_des):
    
    # put it in the Q1 reference frame:
    # rot = np.array([[0,1,0],
                #    [-1,0,0],
                #    [0,0,1]])
    
    pos_vect = np.array([[x_des],
                        [y_des],
                        [1]])

    print(f"Inverse Kinematics: \nPostion vector before rotation: {pos_vect}")
    # pos_vect = rot @ pos_vect
    pos_vect = frame_rot(-90, pos_vect) # -90 degrees to bring from horizontal to vertical. 4th quadrant to 1st

    print(f"postion vector after rotation: {pos_vect}")

    x_des = pos_vect[1]
    y_des = pos_vect[2]
    print(f"position vector = {pos_vect}\nx desired = {x_des}\n y desired = {y_des}\n")
   
    # measured lengths
    length1 = 65 # mm
    length2 = 65 # mm --  I think should be 90mm -- actually bring the end effector position to the desired xy
    length_hand = 90 # mm -- will need to subtract to the desired position? 
   

    # as theta 2 will be negative, we use this version. credit: 
    # https://robotacademy.net.au/lesson/inverse-kinematics-for-a-2-joint-robot-arm-using-geometry/
    q2 = -np.arccos((x_des**2 + y_des**2 - length1**2 - length2**2 ) / (2*length1*length2))
    # atan2 only bc 2 inputs simplifies inputting the equations
    q1 = np.arctan2(x_des,y_des) + np.arctan2(length2*np.sin(q2), (length1+length2*np.cos(q2)))

    # bring back to degrees
    q1 = np.rad2deg(q1)
    q2 = np.rad2deg(q2)

    # negative values for atan would be bad for us. mistake here
    if(q1 < 0):
        print(f"desired theta1 is Negative. bounding to positive value")
        q1 = abs(q1)
    
    
    # now need to rotate -90 degrees -- apply rotation matrix

    # q_v = np.array((x),(y),(1))
    # print(f"angle vectors {q_v}")
    

# assumes the arm is outstretched -- shoulder and elbow both at 180degrees. 
def wrist_map(ping):
    # just because the arm angles are all measured in mm
    x = ping * 10
    max_dist = 90 # just a guess right now. could be 8.5 - 9.5 < ------
    min_dist = 50
    grab_length = 90

    #clamp em
    if x > max_dist:
        x = max_dist
    if x < min_dist:
        x = min_dist

    wrist_th = np.arcsin((x-min_dist)/grab_length)
    wrist_th = np.rad2deg(wrist_th)
    
    # shift + 5
    wrist_th = wrist_th + 5
    print(f"set wrist to:{wrist_th}")
    return wrist_th



if __name__ == '__main__':
    # implementing the DH table for the Arm.
    
    # *** need to be updated *** remeasure!

    base_j = joint(90  , 0 ,  0 , -30)
    # from the base to the shoulder
    #   --shoulder servo angled forward -- to look horizontal
    shoulder_j = joint( -70, 0   , 0 , 45)
    # from the shoulder to the elbow
    #   --elbow servo angled up a bit
    elbow_j = joint(50, 0   , 0 , 65)
    #from elbow to wrist -- should end up about level. 
    #   --wrist angled to the max -- either 180 or 0 idr
    wrist_j = joint(-20, 0, 0  , 65)
    #from wrist to hand
    hand_j = joint(0  , 0   , 0  , 90)


    # Set y to negative idk, 5? tbd with measurments. and the x will be transformed to a y, computed, then transformed back to x.
    # x will be the ping sensor distance. detected. 
    print(f"basic test for forward kinematics 2-link:\n{kin_fwd(15, 45)}\n -- should be ")
    print(f"basic test for inverse kinematics 2-link:\n{kin_inv(80,-10)}\n -- should be ")






