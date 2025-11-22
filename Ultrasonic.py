## UltraSonic Sensors-- HC-SR04
# in venv
# !pip install gpiozero pigpio
    

# in rpi terminal
# sudo apt install gpiozero
# sudo apt-get install python3-pigpio
# sudo systemctl enable pigpiod -- for boot. did not work for me.
# remore GPIO to access from venv -- 

from gpiozero import DistanceSensor
from gpiozero.pins.pigpio import PiGPIOFactory
import time

GLOBAL_ULTRASONIC_INFO = {}
ULTRASONIC_INFO_LOCK = threading.lock()


def Ultrasonic(RPI_IP_ADD, sample_rate, echo_pin, trigger_pin):
        
         # Declare globals here, inside the function, for modification access
        global GLOBAL_TARGET_INFO
        global TARGET_INFO_LOCK
        global LAST_FRAME_STITCHED
        global LAST_FRAME_LOCK
        global ping_dist
        global last_time

# host=RPI_IP_ADD 
        factory = PiGPIOFactory()      # gpio access lib -- set up required for pigpio. -- sudo pigpiod
        last_time = time.time()                       # timestamp
        period = 1/sample_rate                        # time btn publishing samples

        sensor = DistanceSensor(echo_pin, trigger_pin, pin_factory=factory)
        while True:
            if(time.time() - last_time >= period):

                ping_dist = sensor.distance * 100  #cm
                last_time = time.time()
                with TARGET_INFO_LOCK
                    GLOBAL_TARGET_INFO = {'US_distance_cm': ping_dist, 'ts': last_time}
                print(f"Ultrasonic distance: {ping_dist}")
    

# ------------------------------
# THREAD SUPPORT FOR MAIN PROGRAM
# ------------------------------

import threading

def start_thread():
    """Starts Urltrasonic() inside a daemon thread so main.py can use it."""
    global ULTRASONIC_THREAD, STOP_FLAG
    if ULTRASONIC_THREAD is not None and ULTRASONIC_THREAD.is_alive():
        print("[Ultrasonic] Already running.")
        return
    STOP_FLAG = False

    def runner():
        Ultrasonic()  # your original Ultrasonic() loops until STOP_FLAG or 'q'

    ULTRASONIC_THREAD = threading.Thread(target=runner, daemon=True)
    ULTRASONIC_THREAD.start()
    print("[Ultrasonic] Started thread.")


def stop_thread():
    """Signals Ultrasonic() to exit cleanly."""
    global STOP_FLAG
    STOP_FLAG = True
    print("[Ultrasonic] Stop requested.")


if __name__ == '__main__':
    IP = '10.0.0.12'
    sample_rate =10
    echo_pin = 17
    trigger_pin = 27
    Ultrasonic(IP, sample_rate, echo_pin, trigger_pin)
