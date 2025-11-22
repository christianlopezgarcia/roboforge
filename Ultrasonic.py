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


def Ultrasonic(RPI_IP_ADD, sample_rate, echo_pin, trigger_pin):
        
         # Declare globals here, inside the function, for modification access
        global GLOBAL_TARGET_INFO
        global TARGET_INFO_LOCK
        global LAST_FRAME_STITCHED
        global LAST_FRAME_LOCK

        factory = PiGPIOFactory(host=RPI_IP_ADD)      # gpio access lib -- set up required for pigpio. -- sudo pigpiod
        last_time = time.time()                       # timestamp
        period = 1/sample_rate                        # time btn publishing samples

        sensor = DistanceSensor(echo_pin=17, trigger_piner=27, pin_factory=factory)
        while True:
            if(time.time() > last_time):

                self.ping_dist = self.sensor.distance * 100  #cm
                last_time = time.time()

    






# ------------------------------
# THREAD SUPPORT FOR MAIN PROGRAM
# ------------------------------

import threading

def start_thread():
    """Starts run() inside a daemon thread so main.py can use it."""
    global STEREO_THREAD, STOP_FLAG
    if STEREO_THREAD is not None and STEREO_THREAD.is_alive():
        print("[Stereo] Already running.")
        return

    STOP_FLAG = False

    def runner():
        run()  # your original run() loops until STOP_FLAG or 'q'

    STEREO_THREAD = threading.Thread(target=runner, daemon=True)
    STEREO_THREAD.start()
    print("[Stereo] Started thread.")


def stop_thread():
    """Signals run() to exit cleanly."""
    global STOP_FLAG
    STOP_FLAG = True
    print("[Stereo] Stop requested.")


if __name__ == '__main__':
    run()