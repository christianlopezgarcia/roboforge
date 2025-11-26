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
STOP_FLAG = False

# pins is the set of pins for the ultrasonic sensors. each sensor needs: 
#   sensor_number: echo_pin, trigger pin 
def Ultrasonic(sample_rate, echo_pin, trigger_pin):
    global STOP_FLAG

    print("[Ultrasonic] Thread started")

    # Pigpio connection

    last_time = time.time()
    factory = PiGPIOFactory()
    period = 1/sample_rate
    sensor = DistanceSensor(echo_pin, trigger_pin, pin_factory=factory)



    while not STOP_FLAG:
        now = time.time()
        if now - last_time >= period:

            ping_dist = sensor.distance * 100
            last_time = now

            with ULTRASONIC_INFO_LOCK:
                GLOBAL_ULTRASONIC_INFO.clear()
                GLOBAL_ULTRASONIC_INFO.update({
                    'US_distance_cm': ping_dist,
                    'ts': last_time
                })

        time.sleep(0.001)

    sensor.close()
    print("[Ultrasonic] Sensor closed.")

# THREAD CONTROL
ULTRASONIC_THREAD = None

def start_thread(sample_rate, echo_pin, trigger_pin):
    """Starts Urltrasonic() inside a daemon thread so main.py can use it."""
    global ULTRASONIC_THREAD, STOP_FLAG

    if ULTRASONIC_THREAD and ULTRASONIC_THREAD.is_alive():
        print("[Ultrasonic] Already running.")
        return
    STOP_FLAG = False
    ULTRASONIC_THREAD = threading.Thread(
        target=Ultrasonic,
        args=(sample_rate, echo_pin, trigger_pin),
        daemon=True
    )
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
    Ultrasonic(sample_rate, echo_pin, trigger_pin)

    