#!/usr/bin/env python3
"""
Robot arm controller - Hybrid Servo Control
- Uses adafruit_motor.servo for smooth movement on specified joints (Base/Shoulder).
- Uses ServoKit for instantaneous movement on other joints (Elbow/Wrist/Hand).
- Keeps joint & DH helpers present for forward kinematics (get_x_y_z)
- Stores named poses and exposes manual APIs including set_all(dict) and add_angle(name, delta)
"""

import time
import board
import numpy as np
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from adafruit_servokit import ServoKit

# ------------------------------------------------------------
# Joint / DH helpers (kept as requested)
# ------------------------------------------------------------
class joint:
    def __init__(self, theta, d, alpha, a):
        # Store in degrees for theta/alpha for readability (we convert inside A_i)
        self.theta = theta
        self.alpha = alpha
        self.a = a
        self.d = d

    def A(self):
        return A_i(self.theta, self.alpha, self.a, self.d)

    def add_angle(self, theta = 0):
        """Mutate theta (kept for compatibility). Prefer creating copies for FK."""
        self.theta = self.theta + theta
        return self

def A_i(th_i, alpha_i, a_i, d_i):
    """Compute transformation matrix for a single DH row (angles in degrees)."""
    th_r = np.deg2rad(th_i)
    alpha_r = np.deg2rad(alpha_i)
    A = np.array([
        [np.cos(th_r), -np.sin(th_r)*np.cos(alpha_r),  np.sin(th_r)*np.sin(alpha_r), a_i*np.cos(th_r)],
        [np.sin(th_r),  np.cos(th_r)*np.cos(alpha_r), -np.cos(th_r)*np.sin(alpha_r), a_i*np.sin(th_r)],
        [0,             np.sin(alpha_r),              np.cos(alpha_r),              d_i],
        [0,             0,                            0,                            1]
    ])
    return A

def multiply_all(joints_list):
    """Multiply A matrices of each joint in order A1 * A2 * ..."""
    if not joints_list:
        return np.eye(4)
    T = joints_list[0].A()
    for j in joints_list[1:]:
        T = np.matmul(T, j.A())
    return T

def get_carte(T):
    """Return the x,y,z translation vector from a homogenous transform"""
    return T[0:3, 3]

# ------------------------------------------------------------
# Servo & Pose Controller
# ------------------------------------------------------------

# Servo channel mapping (names -> PCA channel numbers)
SERVO_CHANNELS = {
    "base": 6,
    "shoulder": 5,
    "elbow": 4,
    "wrist": 3,
    "hand": 1
}

# Startup / default angles for the named servos (degrees)
STARTUP_POSE = {
    "base": 90,
    "shoulder": 180,
    "elbow": 0,
    "wrist": 0,
    "hand": 180
}

# Movement tuning
MIN_PULSE = 500
MAX_PULSE = 2500
SMOOTH_STEP = 2         # degrees per step for smooth servos
SMOOTH_DELAY = 0.05     # seconds between smooth steps (Adjusted slightly lower from 0.055)

# Which servos should use the custom smooth movement (high-torque / big / metal gear)
SMOOTH_SERVOS = {"shoulder", "base"} 
SMOOTH_CHANNELS = {SERVO_CHANNELS[name] for name in SMOOTH_SERVOS} # Channels of smooth servos

class RobotArm:
    def __init__(self):
        # Initialize I2C and PCA9685
        try:
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(self.i2c)
            self.pca.frequency = 50
        except Exception as e:
            print(f"Error initializing I2C/PCA9685: {e}")
            raise

        # Initialize ServoKit for all channels (used by non-smooth joints)
        self.kit = ServoKit(channels=16, i2c=self.i2c, frequency=50)

        # Initialize adafruit_motor.servo objects for custom smooth movement
        self.smooth_servo_objects = {}
        
        # Pose storage + current tracking
        self.poses = {}
        self.current_angles = {}    # current known/last-set angles

        # Safety limits (min, max) per joint — edit to match your hardware's mechanical limits
        self.limits = {
            "base":      (0, 180),
            "shoulder":  (20, 170),
            "elbow":     (0, 180),
            "wrist":     (0, 180),
            "hand":      (0, 180)
        }
        
        # Setup servos
        for name, ch in SERVO_CHANNELS.items():
            start_angle = int(STARTUP_POSE.get(name, 90))
            safe_start = self.clamp_angle(name, start_angle)
            
            # 1. Setup PWM pulse range for ALL channels via ServoKit
            # This is done on the ServoKit object, even if we use adafruit_motor for control
            self.kit.servo[ch].set_pulse_width_range(MIN_PULSE, MAX_PULSE)

            if ch in SMOOTH_CHANNELS:
                # 2a. Create adafruit_motor.servo object for smooth joints
                s = servo.Servo(self.pca.channels[ch], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE)
                self.smooth_servo_objects[name] = s
                # Set initial angle using adafruit_motor
                s.angle = safe_start
            else:
                # 2b. Set initial angle using ServoKit for non-smooth joints
                self.kit.servo[ch].angle = safe_start
            
            self.current_angles[name] = safe_start
            time.sleep(0.02) # Small delay for stability

        # Keep the DH base "templates" (nominal DH joint definitions)
        self.ping2base_j       = joint(180,  0,    0, 45)
        self.base2shoulder_j   = joint(-90,  0,    0, 70)
        self.shoulder2elbow_j  = joint(90,   0,    0, 65)
        self.elbow2wrist_j     = joint(57,   0,    0, 65)
        self.wrist2effector_j  = joint(-64,  0,    0, 90)

        # Add default poses
        self.add_pose("default", **STARTUP_POSE)
        self.add_pose("45_down", base=0, shoulder=90, elbow=180, wrist=0, hand=180)
        self.add_pose("safe", base=0, shoulder=180, elbow=0, wrist=180, hand=180)
        self.add_pose("straight_up", base=0, shoulder=90, elbow=90, wrist=90, hand=90)
        self.add_pose("reach_forward", base=90, shoulder=180, elbow=85, wrist=90, hand=180)
        self.add_pose("elbow_L", base=0, shoulder=90, elbow=180, wrist=90, hand=180)
        self.add_pose("wrist_L", base=0, shoulder=90, elbow=90, wrist=0, hand=180)

        print("RobotArm ready. Hybrid control active. Startup pose applied.")
    
    # ---------------------
    # Safety helpers
    # ---------------------
    def clamp_angle(self, name, angle):
        """Clamp angle to the joint's safety limits and return int."""
        if name not in self.limits:
            # if no limits defined, fallback to 0-180
            mn, mx = 0, 180
        else:
            mn, mx = self.limits[name]
        return int(max(mn, min(mx, int(round(angle)))))
    
    # ---------------------
    # Movement helpers
    # ---------------------

    def _move_smooth_custom(self, name, target, step=SMOOTH_STEP, delay=SMOOTH_DELAY):
        """
        Move a servo smoothly using the custom adafruit_motor.servo object.
        This function implements the working sweep logic from the user's demo.
        """
        target = self.clamp_angle(name, target)
        current = self.current_angles.get(name)
        s = self.smooth_servo_objects[name]
        print(s.__dict__)
        if target == current:
            return

        direction = 1 if target > current else -1

        # Iterate in steps, clamping each step
        for a in range(current, target + direction, direction * step):
            print(a)
            s.angle = a
            self.current_angles[name] = a
            time.sleep(delay)

        # Final assignment to ensure target is reached (clamped)
        final_target = self.clamp_angle(name, target)
        s.angle = final_target
        self.current_angles[name] = final_target


    def _set_angle_instantly(self, name, angle):
        """Set servo to angle immediately (with clamp) using ServoKit."""
        a = self.clamp_angle(name, angle)
        ch = SERVO_CHANNELS[name]
        self.kit.servo[ch].angle = a
        self.current_angles[name] = a

    def _smooth_move(self, name, target):
        """Move a servo smoothly if configured (custom logic); otherwise, set instantly (ServoKit)."""
        target = self.clamp_angle(name, target)
        
        if name in SMOOTH_SERVOS:
            # Use custom smooth movement for configured joints
            self._move_smooth_custom(name, target)
        else:
            # Use instantaneous ServoKit movement for other joints
            self._set_angle_instantly(name, target)

    # ---------------------
    # Pose management
    # ---------------------
    def add_pose(self, name, base=90, shoulder=90, elbow=90, wrist=90, hand=90):
        """Save a named pose (angles will be clamped only when applied)."""
        self.poses[name] = {
            "base": int(base),
            "shoulder": int(shoulder),
            "elbow": int(elbow),
            "wrist": int(wrist),
            "hand": int(hand)
        }
        print(f"[POSE] Saved '{name}': {self.poses[name]}")

    def move_to_pose(self, name):
        """Move robot to a previously defined pose name (applies safety clamping)."""
        if name not in self.poses:
            raise KeyError(f"Pose '{name}' not defined")

        pose = self.poses[name]
        print(f"\n→ Moving to pose '{name}': {pose}")

        # Use a sensible order: base -> shoulder -> elbow -> wrist -> hand
        order = ["base", "shoulder", "elbow", "wrist", "hand"]
        for joint_name in order:
            if joint_name in pose:
                target = pose[joint_name]
                self._smooth_move(joint_name, target)
                time.sleep(0.01)

        print("✓ Pose reached.")

    # ---------------------
    # Manual APIs
    # ---------------------
    def set_joint(self, name, angle):
        """Directly set one joint (clamped, smooth or instant depending on joint)."""
        if name not in SERVO_CHANNELS:
            raise KeyError(f"Unknown joint '{name}'")
        self._smooth_move(name, angle)

    def add_angle(self, name, delta):
        """Add delta degrees to a joint (positive or negative)."""
        if name not in SERVO_CHANNELS:
            raise KeyError(f"Unknown joint '{name}'")
        current = int(self.current_angles.get(name, 90))
        target = current + int(delta)
        self._smooth_move(name, target)

    def set_all(self, pose=None, **kwargs):
        """
        Set multiple joints from a dictionary or kwargs.
        Only joints included are changed. All values are clamped to safety limits.
        """
        commands = pose.copy() if pose else {}
        commands.update(kwargs)

        # Move in stable order; apply smoothing rules per joint
        order = ["base", "shoulder", "elbow", "wrist", "hand"]
        for joint_name in order:
            if joint_name in commands:
                try:
                    self._smooth_move(joint_name, commands[joint_name])
                except Exception as e:
                    print(f"[ERROR] setting {joint_name}: {e}")

    # Convenience grab/release that obey limits
    def grab(self, strength=120):
        """Move hand to strength (clamped)."""
        self._smooth_move("hand", strength)

    def release(self):
        """Open hand to its safe max (clamped)."""
        mn, mx = self.limits["hand"]
        self._smooth_move("hand", mx)

    # ---------------------
    # Forward kinematics
    # ---------------------
    def get_x_y_z(self):
        """
        Build a temporary set of joints from the stored DH templates and the current servo angles.
        Returns (x, y, z). Sign mapping is heuristically based on your earlier logic.
        """
        base_a = int(self.current_angles.get("base", 90))
        shoulder_a = int(self.current_angles.get("shoulder", 180))
        elbow_a = int(self.current_angles.get("elbow", 0))
        wrist_a = int(self.current_angles.get("wrist", 0))

        # Template thetas:
        t_ping2base = self.ping2base_j.theta
        t_base2shoulder = self.base2shoulder_j.theta
        t_shoulder2elbow = self.shoulder2elbow_j.theta
        t_elbow2wrist = self.elbow2wrist_j.theta
        t_wrist2eff = self.wrist2effector_j.theta

        # Apply simple mapping of servo angles -> joint thetas (calibrate later)
        j_ping2base = joint(t_ping2base + (base_a - 90), self.ping2base_j.d, self.ping2base_j.alpha, self.ping2base_j.a)
        j_base2shoulder = joint(t_base2shoulder, self.base2shoulder_j.d, self.base2shoulder_j.alpha, self.base2shoulder_j.a)
        j_shoulder2elbow = joint(t_shoulder2elbow - shoulder_a, self.shoulder2elbow_j.d, self.shoulder2elbow_j.alpha, self.shoulder2elbow_j.a)
        j_elbow2wrist = joint(t_elbow2wrist - elbow_a, self.elbow2wrist_j.d, self.elbow2wrist_j.alpha, self.elbow2wrist_j.a)
        j_wrist2effector = joint(t_wrist2eff + wrist_a, self.wrist2effector_j.d, self.wrist2effector_j.alpha, self.wrist2effector_j.a)

        all_joints = [j_ping2base, j_base2shoulder, j_shoulder2elbow, j_elbow2wrist, j_wrist2effector]
        T06 = multiply_all(all_joints)
        xyz = get_carte(T06)
        return tuple(xyz)

    # Debug helper
    def status(self):
        print("Current angles:", self.current_angles)
        try:
            xyz = self.get_x_y_z()
            print("FK end-effector (x,y,z):", tuple(round(i, 2) for i in xyz))
        except Exception as e:
            print("FK compute error:", e)

    # Optional: safe shutdown / stop PWM (if needed)
    def teardown(self):
        try:
            print("Disabling all PWM outputs...")
            # For adafruit_motor.servo objects (smooth servos)
            for s in self.smooth_servo_objects.values():
                s.angle = None 
            # For ServoKit objects (other servos)
            for ch in SERVO_CHANNELS.values():
                 if SERVO_CHANNELS.get(ch) not in SMOOTH_CHANNELS:
                     self.kit.servo[ch].angle = None
            
            # De-init the PCA9685
            self.pca.deinit()
        except Exception as e:
            print(f"Teardown error: {e}")

# ------------------------------------------------------------
# Demo usage
# ------------------------------------------------------------
if __name__ == "__main__":
    arm = RobotArm()
    time.sleep(0.5)

    print('------ DEFAULT --------')
    arm.move_to_pose("default")
    time.sleep(0.5)
    arm.status()
    
    print('\n------ reach_forward --------')
    arm.move_to_pose("reach_forward")
    time.sleep(0.6)
    arm.status()

    # set_all with a dict (only the included joints are changed)
    print('\n------ set_all (hybrid) --------')
    # This will use custom smooth move for base/shoulder and ServoKit instant for elbow/wrist/hand
    arm.set_all({'base': 180, 'shoulder': 90, 'elbow': 0, 'wrist': 0, 'hand': 180})
    time.sleep(0.6)
    arm.status()

    # # Demonstration of manual APIs using the hybrid approach
    # print('\n------ add_angle elbow (instant) --------')
    # arm.add_angle("elbow", -10)
    # time.sleep(0.5)
    # arm.status()

    # print('\n------ add_angle base (smooth) --------')
    # arm.add_angle("base", -20)
    # time.sleep(0.5)
    # arm.status()

    # print('\n------ grab 40------')
    # arm.grab(40)
    # time.sleep(0.5)
    # print('\n ------45_down------')
    # arm.move_to_pose("45_down")
    # time.sleep(0.6)
    # arm.status()

    # print('\n ------release------')
    # arm.release()
    # time.sleep(0.5)

    print('\n ------Default------')
    arm.move_to_pose("default")
    arm.status()

