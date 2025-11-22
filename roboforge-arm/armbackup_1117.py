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
        "base": 90,
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

    POSES = {
        "default":      {"base": 90, "shoulder": 180, "elbow": 0,  "wrist": 0,  "hand": 180},
        "reach_forward":{"base": 0,  "shoulder": 180, "elbow": 85, "wrist": 90, "hand": 180},
        "45_down":      {"base": 0,  "shoulder": 90,  "elbow": 180,"wrist": 0,  "hand": 180},
        "safe":         {"base": 0,  "shoulder": 180, "elbow": 0,  "wrist": 180,"hand": 180},
        "straight_up":  {"base": 0,  "shoulder": 90,  "elbow": 90, "wrist": 90,"hand": 90},
        "elbow_L":      {"base": 0,  "shoulder": 90,  "elbow": 180,"wrist": 90,"hand": 180},
        "wrist_L":      {"base": 0,  "shoulder": 90,  "elbow": 90, "wrist": 0,"hand": 180},
    }

    MIN_PULSE = 500
    MAX_PULSE = 2500
    STEP = 2
    DELAY = 0.05

    def __init__(self):

        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c)
        self.pca.frequency = 50
        self.kit = ServoKit(channels=16, i2c=self.i2c)

        self.current = {}
        self.large = {}

        # initialize all servos
        for name, ch in self.CHANNELS.items():
            angle = self.clamp(name, self.STARTUP[name])

            if name in self.LARGE:
                s = servo.Servo(
                    self.pca.channels[ch],
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
        self.kit.servo[ch].angle = target
        self.current[name] = float(self.kit.servo[ch].angle)  # REAL angle

    def add_angle(self, name, delta):
        self.set_joint(name, self.current[name] + delta)

    def set_all(self, mapping):
        for j in ["base", "shoulder", "elbow", "wrist", "hand"]:
            if j in mapping:
                self.set_joint(j, mapping[j])

    def move_to_pose(self, name):
        print(f"Pose: {name}")
        self.set_all(self.POSES[name])

    def status(self):
        print("Angles:", self.current)

    # ------------------------------------------------------------
    # Move using inverse kinematics with YOUR real robot lengths
    # ------------------------------------------------------------
    def move_xyz(self, x, y, z):
        """
        Compute base, shoulder, elbow, and wrist angles
        for a (x, y, z) target in cm.
        Uses 3-link IK with link lengths measured from your robot.
        """

        # Your real robot link lengths (cm)
        L1 = 7.0   # shoulder → elbow
        L2 = 6.0   # elbow → wrist
        L3 = 4.5   # wrist → gripper offset

        # -----------------------------------------
        # 1) BASE rotation (simple planar rotation)
        # -----------------------------------------
        base_angle = math.degrees(math.atan2(y, x))

        # Distance from base origin in horizontal plane
        r = math.sqrt(x**2 + y**2)

        # Vertical coordinate
        h = z

        # Subtract the wrist offset L3 from r and h
        # So IK computes elbow pointing to the wrist joint, not the gripper tip
        wx = r - L3   # projected x-distance to wrist pivot
        wz = h        # z stays the same (offset is horizontal only)

        # ------------------------------
        # 2) Distance from shoulder → wrist
        # ------------------------------
        d = math.sqrt(wx**2 + wz**2)

        # Check if reachable
        if d > (L1 + L2):
            raise ValueError(f"Target ({x},{y},{z}) out of reach")

        # ------------------------------
        # 3) SHOULDER ANGLE
        # ------------------------------

        theta = math.degrees(math.atan2(wz, wx))

        # Law of cosines (inner angle at shoulder)
        cos_a = (L1**2 + d**2 - L2**2) / (2 * L1 * d)
        a = math.degrees(math.acos(cos_a))

        shoulder_angle = theta + a  # typical forward arm pose

        # ------------------------------
        # 4) ELBOW ANGLE
        # ------------------------------

        cos_b = (L1**2 + L2**2 - d**2) / (2 * L1 * L2)
        b = math.degrees(math.acos(cos_b))

        elbow_angle = 180 - b   # servo geometry (folds inward)

        # ------------------------------
        # 5) WRIST ANGLE (keep tool vertical)
        # ------------------------------
        wrist_angle = 180 - (shoulder_angle + elbow_angle)

        # ------------------------------
        # 6) Execute the arm movement
        # ------------------------------
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

    print('\n------ DEFAULT --------')
    arm.move_to_pose("default")
    time.sleep(0.5)
    arm.status()

    print('\n------ set_all (hybrid) --------')
    arm.set_all({'base': 180, 'shoulder':90, 'elbow': 0, 'wrist': 0, 'hand': 180})
    time.sleep(0.6)
    arm.status()

    print('\n ------Default------')
    arm.move_to_pose("default")
    arm.status()

    # print("\n------ Move to XYZ (10, 5, 8) ------")
    # result = arm.move_xyz(5, 0, 0)
    # print("IK Angles:", result)
    # arm.status()

    print('\n ------Default------')
    arm.move_to_pose("default")
    arm.status()
