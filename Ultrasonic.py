## UltraSonic Sensors-- HC-SR04
# in venv
# !pip install gpiozero pigpio
    

# in rpi terminal
# sudo apt install gpiozero
# sudo apt-get install python3-pigpio
# sudo systemctl enable pigpiod -- for boot. did not work for me.
# remore GPIO to access from venv -- 
import threading 
from gpiozero import DistanceSensor
from gpiozero.pins.pigpio import PiGPIOFactory
import time

GLOBAL_ULTRASONIC_INFO = {} 
ULTRASONIC_INFO_LOCK = threading.Lock()

# pins is the set of pins for the ultrasonic sensors. each sensor needs: 
#   sensor_number: echo_pin, trigger pin 
def Ultrasonic(sample_rate, echo_pin, trigger_pin):
        
         # Declare globals here, inside the function, for modification access
        global GLOBAL_ULTRASONIC_INFO
        global ULTRASONIC_INFO_LOCK
        global ping_dist
        global last_time
        last_time= time.time()                       # timestamp
        RPI_IP_ADD = 0
        factory = PiGPIOFactory()      # gpio access lib -- set up required for pigpio. -- sudo pigpiod
        period = 1/sample_rate                        # time btn publishing samples
        
        sensor = DistanceSensor(echo_pin, trigger_pin, pin_factory=factory)
        while True:                
            if(time.time() - last_time >= period):

                ping_dist = sensor.distance * 100  #cm
                last_time = time.time()

                with ULTRASONIC_INFO_LOCK:        
                    GLOBAL_ULTRASONIC_INFO = {'US_distance_cm': ping_dist, 'ts': last_time}
                
                # return ping_dist
                print(f"Ultrasonic distance: {ping_dist}")
    

# ------------------------------
# THREAD SUPPORT FOR MAIN PROGRAM
# ------------------------------

import threading

def start_thread(sample_rate, echo_pin, trigger_pin):
    """Starts Urltrasonic() inside a daemon thread so main.py can use it."""
    global ULTRASONIC_THREAD, STOP_FLAG
    if ULTRASONIC_THREAD is not None and ULTRASONIC_THREAD.is_alive():
        print("[Ultrasonic] Already running.")
        return
    STOP_FLAG = False

    def runner(sample_rate, echo_pin, trigger_pin):
        Ultrasonic(sample_rate, echo_pin, trigger_pin)  # your original Ultrasonic() loops until STOP_FLAG or 'q'

    ULTRASONIC_THREAD = threading.Thread(target=runner,args=(sample_rate, echo_pin, trigger_pin), daemon=True)
    ULTRASONIC_THREAD.start()
    print("[Ultrasonic] Started thread.")


def stop_thread():
    """Signals Ultrasonic() to exit cleanly."""
    global STOP_FLAG
    STOP_FLAG = True
    print("[Ultrasonic] Stop requested.")


if __name__ == '__main__':
    IP = '10.0.0.12'
    sample_rate = 10 
    echo_pin = 17   
    trigger_pin = 27
    Ultrasonic(IP, sample_rate, echo_pin, trigger_pin)

    