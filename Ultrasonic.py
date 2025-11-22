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

class Ultrasonic:   
    def __INIT__(self, RPI_IP_ADD, sample_rate = 10, )
        
        # self,RPI_IP_ADD = '10.0.0.53' # T- hostname?
        self.running = False
        self.thread = None
        
        self.factory = PiGPIOFactory(host=self.RPI_IP_ADD)      # gpio access lib -- set up required for pigpio. -- sudo pigpiod
        last_time = time.time()                       # timestamp
        period = 1/sample_rate                        # time btn publishing samples

        self.sensor = DistanceSensor(echo=17, trigger=27, pin_factory=self.factory)
        while True:
            if(time.time() > last_time):
                
                self.ping_dist = self.sensor.distance * 100  #cm
                last_time = time.time()



    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        
    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            if not self.buffer.empty():
                try: self.buffer.get_nowait()
                except queue.Empty: pass
            self.buffer.put(frame)

    def run():



        

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