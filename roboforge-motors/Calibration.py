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
        """
        Interactively calibrate the BNO055 and save the result to disk.

        poll_interval: seconds between status checks/prints
        timeout: max time in seconds before giving up (None = no timeout)

        Returns True if calibration reached (3,3,3,3) and was saved, False otherwise.
        """
        print("=== BNO055 Calibration ===")
        print("Move the sensor through different orientations.")
        print("You want all fields to reach 3 (SYS, Gyro, Accel, Mag).")
        print("Press Ctrl+C to abort.\n")

        start = time.time()

        try:
            while True:
                cal = self.sensor.calibration_status  # (sys, gyro, accel, mag)
                if cal is None or len(cal) != 4:
                    print("Calibration status not available, retrying...")
                    time.sleep(5)
                    continue

                sys_c, gyro_c, accel_c, mag_c = cal
                print(f"CAL -> SYS:{sys_c} G:{gyro_c} A:{accel_c} M:{mag_c}", end="\r")

                # Fully calibrated?
                if sys_c == 3 and gyro_c == 3 and accel_c == 3 and mag_c == 3:
                    print("\n✅ Sensor fully calibrated!")
                    self.save_calibration()
                    return True

                # Timeout?
                #if timeout is not None and (time.time() - start) > timeout:
                #    print("\n⏱️ Calibration timed out before reaching (3,3,3,3).")
                #    return False

                time.sleep(5)

        except KeyboardInterrupt:
            print("\nCalibration aborted by user.")
            return False



    
if __name__ == "__main__":
    
    #Create I2C Objects
    i2c = busio.I2C(board.SCL, board.SDA)
    bno = BNO055(i2c)
    
    bno.calibrate_sensor
   
    print("done")  
