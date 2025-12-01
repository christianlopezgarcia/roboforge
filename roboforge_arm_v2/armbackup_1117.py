import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from adafruit_servokit import ServoKit
import math


class RobotArm:

    CHANNELS = {
        "base": 6,
        "shoulder": 5,
        "elbow": 4,
        "wrist": 3,
        "hand": 1
    }

    LARGE = {"base", "shoulder"}
    SMALL = {"elbow", "wrist", "hand"}

    STARTUP = {
        "base": 105,
        "shoulder": 180,
        "elbow": 0,
        "wrist": 0,
        "hand": 180
    }

    LIMITS = {
        "base": (0, 180),
        "shoulder": (0, 180),
        "elbow": (0, 180),
        "wrist": (0, 180),
        "hand": (0, 180)
    }

    OPEN = 160
    CLOSED = 25 #20 gave wierd issues good luck kevin
    POSES = {
        "default":             {"base": 90, "shoulder": 175, "elbow": 0,  "wrist": 0,  "hand": OPEN}, #solid
        "default_closed":      {"base": 90, "shoulder": 180, "elbow": 0,  "wrist": 0,  "hand": CLOSED},

        "focused":              {"base": 90, "shoulder": 180, "elbow": 80,  "wrist":5 ,  "hand": OPEN}, #solid
        "focused_closed":       {"base": 90, "shoulder": 180, "elbow": 80,  "wrist": 5,  "hand": CLOSED}, #solid
        
        "5cm":                  {"base": 90, "shoulder": 180, "elbow": 80,  "wrist": 5,  "hand": OPEN}, #solid
        "5cm_closed":           {"base": 90, "shoulder": 180, "elbow": 80,  "wrist": 5,  "hand": CLOSED}, #solid
        "5cm_30r_closed":       {"base": 60, "shoulder": 180, "elbow": 80,  "wrist": 5,  "hand": CLOSED}, 

        "8cm":                  {"base": 90, "shoulder": 180, "elbow": 80,  "wrist": 25,  "hand": OPEN}, #solid
        "8cm_closed":           {"base": 90, "shoulder": 180, "elbow": 80,  "wrist": 25,  "hand": CLOSED}, #solid
        "8cm_30r_closed":       {"base": 60, "shoulder": 180, "elbow": 80,  "wrist": 25,  "hand": CLOSED}, 

        "10_cm":                {"base": 90,  "shoulder": 200, "elbow": 85, "wrist": 45, "hand": OPEN}, #solid
        "10_cm_closed":         {"base": 90,  "shoulder": 200, "elbow": 85, "wrist": 45, "hand": CLOSED}, #solid
        "10_cm_30r_closed":     {"base": 60,  "shoulder": 200, "elbow": 85, "wrist": 45, "hand": CLOSED}, #solid

        
        "fold_over":        {"base": 90, "shoulder": 90, "elbow": 0,  "wrist": 90,  "hand": CLOSED},
        "fold_over_open":   {"base": 90, "shoulder": 90, "elbow": 0,  "wrist": 90,  "hand": OPEN},
        
        "wide_view":        {"base": 90, "shoulder": 180, "elbow": 15,  "wrist": 0,  "hand": 180}, #solid
        "narrow_view":      {"base": 90,  "shoulder": 90,  "elbow": 150,"wrist": 45,  "hand": 180}, #solid

        "21_84_cm_view":    {"base": 90, "shoulder": 180, "elbow": 15,  "wrist": 0,  "hand": 180}, #solid
        "4_21_view":        {"base": 90,  "shoulder": 50,  "elbow": 160,"wrist": 0,  "hand": 180}, #solid

        "45_down":      {"base": 90,  "shoulder": 90,  "elbow": 100,"wrist": 0,  "hand": 180},
        "reach_forward":{"base": 90,  "shoulder": 180, "elbow": 85, "wrist": 90, "hand": 180},
        "safe":         {"base": 90,  "shoulder": 180, "elbow": 0,  "wrist": 180,"hand": 180},
        "straight_up":  {"base": 90,  "shoulder": 90,  "elbow": 90, "wrist": 90,"hand": 90},
        "elbow_L":      {"base": 90,  "shoulder": 90,  "elbow": 180,"wrist": 90,"hand": 180},
        "wrist_L":      {"base": 90,  "shoulder": 90,  "elbow": 100,"wrist": 0,  "hand": 180}, #solid
    }

    MIN_PULSE = 500
    MAX_PULSE = 3000
    STEP = 2
    FAST_STEP = 2
    DELAY = 0.05
    ACTUATION_RANGE = 200

    def __init__(self, i2c_obj):

        self.i2c = i2c_obj
        self.pca = PCA9685(self.i2c)
        self.pca.frequency = 50
        self.kit = ServoKit(channels=16, i2c=self.i2c)

        self.current = {}
        self.large = {}
        self.current_pose = None

        # initialize all servos
        for name, ch in self.CHANNELS.items():
            angle = self.clamp(name, self.STARTUP[name])

            if name in self.LARGE:
                s = servo.Servo(
                    self.pca.channels[ch],
                    actuation_range = 190, #Actuation Large servo only. default 180. maps onto min/max pulses < - - -
                    min_pulse=self.MIN_PULSE,
                    max_pulse=self.MAX_PULSE
                )
                self.large[name] = s
                s.angle = angle
                self.current[name] = s.angle  # REAL angle

            else:
                self.kit.servo[ch].set_pulse_width_range(self.MIN_PULSE, self.MAX_PULSE)
                self.kit.servo[ch].angle = angle
                self.current[name] = self.kit.servo[ch].angle  # REAL angle

            time.sleep(0.05)

        print("RobotArm ready (angle-safe mode).")

    def clamp(self, joint, angle):
        lo, hi = self.LIMITS[joint]
        return max(lo, min(hi, int(angle)))

    # ------------------------------------------------------------
    # MOVEMENT (fixed so angles NEVER desync)
    # ------------------------------------------------------------
    def set_joint(self, name, target):

        target = self.clamp(name, target)

        # ---- BIG SERVOS (smooth) ----
        if name in self.LARGE:
            s = self.large[name]
            current = float(s.angle)

            if current is None:
                current = self.current[name]  # fallback

            direction = 1 if target > current else -1

            # fixed stepping that never skips the final target
            a = current
            while (direction == 1 and a < target) or (direction == -1 and a > target):
                a += direction * self.STEP
                if (direction == 1 and a > target) or (direction == -1 and a < target):
                    a = target
                s.angle = a
                time.sleep(self.DELAY)

            self.current[name] = float(s.angle)  # read back REAL angle
            return

        # ---- SMALL SERVOS (instant) ----

        ch = self.CHANNELS[name]
        current = float(self.kit.servo[ch].angle)
        # self.kit.servo[ch].angle = target
        direction = 1 if target > current else -1

        # fixed stepping that never skips the final target
        a = current
        while (direction == 1 and a < target) or (direction == -1 and a > target):
            a += direction * self.FAST_STEP
            if (direction == 1 and a > target) or (direction == -1 and a < target):
                a = target
            self.kit.servo[ch].angle = a
            time.sleep(.01)

        self.current[name] = float(self.kit.servo[ch].angle)  # REAL angle

    def add_angle(self, name, delta):
        self.set_joint(name, self.current[name] + delta)

    def set_all(self, mapping, reverse=False, start_joint="base"):
        order = ["base", "shoulder", "elbow", "wrist", "hand"]

        # move start_joint to front
        if start_joint in order:
            idx = order.index(start_joint)
            order = order[idx:] + order[:idx]

        if reverse:
            order = order[::-1]

        for joint in order:
            if joint in mapping:
                self.set_joint(joint, mapping[joint])

    def move_to_pose(self, name, reverse = False, start_joint = "base"):
        print(f"Pose: {name}")
        self.current_pose = name
        # if name == "default":
        #     # Step 1 — tuck small joints instantly
        #     self.set_joint("hand", 180)
        #     self.set_joint("wrist", 90)

        #     time.sleep(0.2)

        #     # Step 2 — tuck elbow (instant)
        #     self.set_joint("elbow", 0)
        #     time.sleep(0.2)

        #     # Step 3 — now move big sweeping joints
        #     self.set_joint("shoulder", 180)
        #     self.set_joint("base", 90)

        #     # self.status()
        #     return 
        self.set_all(self.POSES[name], reverse, start_joint)

    def status(self):
        print("Angles:", self.current)

    # ------------------------------------------------------------
    # Move using inverse kinematics with YOUR real robot lengths
    # ------------------------------------------------------------
    def move_xyz(self, x, y, z):
        """
        Compute base, shoulder, elbow, and wrist angles
        for a (x, y, z) target in cm.
        Uses 3-link IK with measured link lengths.
        """

        # Your real robot link lengths (cm)
        L1 = 6.5   # shoulder → elbow
        L2 = 6.5   # elbow → wrist
        L3 = 5.0   # wrist → gripper offset

        # -----------------------------
        # 1) BASE rotation (planar)
        # -----------------------------
        base_angle = math.degrees(math.atan2(y, x))

        # Horizontal distance from base origin
        r = math.sqrt(x**2 + y**2)
        h = z  # vertical distance

        # Offset for wrist (so IK solves for wrist, not gripper tip)
        wx = r - L3
        wz = h

        # -----------------------------
        # 2) Distance from shoulder → wrist
        # -----------------------------
        d = math.sqrt(wx**2 + wz**2)

        # Check if target is reachable
        if d > (L1 + L2):
            raise ValueError(f"Target ({x},{y},{z}) out of reach")

        # -----------------------------
        # 3) SHOULDER angle
        # -----------------------------
        theta = math.degrees(math.atan2(wz, wx))
        cos_a = (L1**2 + d**2 - L2**2) / (2 * L1 * d)
        a = math.degrees(math.acos(cos_a))
        shoulder_angle = theta + a  # typical forward arm pose

        # -----------------------------
        # 4) ELBOW angle
        # -----------------------------
        cos_b = (L1**2 + L2**2 - d**2) / (2 * L1 * L2)
        b = math.degrees(math.acos(cos_b))
        elbow_angle = 180 - b  # servo geometry

        # -----------------------------
        # 5) WRIST angle (keep gripper vertical)
        # -----------------------------
        wrist_angle = 180 - (shoulder_angle + elbow_angle)

        # -----------------------------
        # 6) Execute the arm movement
        # -----------------------------
        self.set_all({
            "base": base_angle,
            "shoulder": shoulder_angle,
            "elbow": elbow_angle,
            "wrist": wrist_angle
        })

        return {
            "base": base_angle,
            "shoulder": shoulder_angle,
            "elbow": elbow_angle,
            "wrist": wrist_angle
        }

def five_cm_pickup():
    arm.move_to_pose("5cm", reverse=True)
    time.sleep(1)
    arm.move_to_pose("5cm_closed")
    time.sleep(1)
    arm.move_to_pose("5cm_30r_closed")
    time.sleep(1)
    arm.move_to_pose("default_closed", reverse=True)
    time.sleep(1)
    arm.move_to_pose("fold_over", reverse=True)
    time.sleep(1)
    arm.move_to_pose("fold_over_open")
    time.sleep(1)
    arm.move_to_pose("4_21_view",reverse = True)


def ten_cm_pickup():
    arm.move_to_pose("10_cm", reverse=True)
    time.sleep(1)
    arm.move_to_pose("10_cm_closed")
    time.sleep(1)
    arm.move_to_pose("10_cm_30r_closed")
    time.sleep(1)
    arm.move_to_pose("fold_over", reverse=True)
    time.sleep(1)
    arm.move_to_pose("fold_over_open")
    time.sleep(1)
    arm.move_to_pose("4_21_view",reverse = True)

    # time.sleep(1)
    # arm.move_to_pose("default")

def eight_cm_pickup():
    arm.move_to_pose("8cm", reverse=True)
    time.sleep(1)
    arm.move_to_pose("8cm_closed")
    time.sleep(1)
    arm.move_to_pose("8cm_30r_closed")
    time.sleep(1)
    arm.move_to_pose("fold_over", reverse=True)
    time.sleep(1)
    arm.move_to_pose("fold_over_open")
    time.sleep(1)
    arm.move_to_pose("4_21_view",reverse = True)
# ------------------------------------------------------------
# Demo usage
# ------------------------------------------------------------
if __name__ == "__main__":
    i2c_obj = busio.I2C(board.SCL, board.SDA)
    arm = RobotArm(i2c_obj)
    time.sleep(0.5)

    print('------ DEFAULT --------')
    arm.move_to_pose("default", reverse= True)
    time.sleep(0.5)
    arm.status()

    ten_cm_pickup()
    five_cm_pickup()
    
    # print('\n------ reach_forward --------')
    # arm.move_to_pose("reach_forward")
    # time.sleep(2)
    # arm.status()

    # print('\n------ DEFAULT --------')
    # arm.move_to_pose("default")
    # time.sleep(0.5)
    # arm.status()

    # print('\n------ set_all (hybrid) --------')
    # arm.set_all({"base": 105,  "shoulder": 90,  "elbow": 150,"wrist": 0,  "hand": 180}) #cool up right pose WALLE mode
    # time.sleep(15)
    # # arm.set_all({"hand": 180,"wrist": 0,"elbow": 0,"shoulder": 180, "base": 90,})
    # # time.sleep(2)
    # arm.status()

    # print('\n ------Default------')
    # arm.move_to_pose("wide_view")
    # time.sleep(4)
    # arm.status()
    # arm.move_to_pose("4_21_view")
    # time.sleep(4)
    # arm.status()
    
    # arm.move_to_pose("5cm")
    # time.sleep(2)
    # arm.status()
    # arm.move_to_pose("5cm_closed")
    # time.sleep(2)
    # arm.status()
    # arm.move_to_pose("5cm_30r_closed")
    # time.sleep(2)
    # arm.status()
    # arm.move_to_pose("default_closed",reverse =True)
    # time.sleep(2)
    # arm.status()
    # arm.move_to_pose("fold_over")
    # time.sleep(2)
    # arm.status()
    # arm.move_to_pose("fold_over_open")
    # time.sleep(2)
    # arm.status()

    # print("\n------ Move to XYZ (10, 5, 8) ------")
    # result = arm.move_xyz(10, 0, -6)
    # time.sleep(0.6)
    # print("IK Angles:", result)
    # arm.status()

    # print('\n ------Default------')
    # arm.move_to_pose("default")
    # arm.status()

    # arm.move_to_pose("10_cm")
    # time.sleep(1)
    # arm.move_to_pose("10_cm_closed")
    # time.sleep(1)
    # arm.move_to_pose("10_cm_30r_closed")
    # time.sleep(1)
    # arm.move_to_pose("fold_over",reverse =True)
    # time.sleep(2)
    # arm.status()
    # arm.move_to_pose("fold_over_open")
    # time.sleep(2)
    # arm.move_to_pose("default")
    # arm.status()

    # print('\n ------Default------')
    # arm.move_to_pose("4_21_view")
    # arm.status()

    # arm.move_to_pose("8cm",reverse = True)
    # time.sleep(1)
    # arm.move_to_pose("8cm_closed")
    # time.sleep(1)
    # arm.move_to_pose("8cm_30r_closed")
    # time.sleep(1)
    # arm.move_to_pose("fold_over",reverse =True)
    # time.sleep(2)
    # arm.status()
    # arm.move_to_pose("fold_over_open")
    # time.sleep(2)
    # arm.move_to_pose("default")
    # arm.status()
    # arm.set_all({"base": 90,  "shoulder": 50,  "elbow": 160,"wrist": 0,  "hand": 180})
    arm.move_to_pose(name = "4_21_view",reverse = True)
    arm.status()


