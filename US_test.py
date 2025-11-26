from Ultrasonic import (
    start_thread,
    stop_thread,
    GLOBAL_ULTRASONIC_INFO,
    ULTRASONIC_INFO_LOCK
)
import time


if __name__ == '__main__':
    
    sample_rate = 100 
    echo_pin = 17   
    trigger_pin = 27
    
    start_thread(sample_rate, echo_pin, trigger_pin)
    
    start_time = time.time()
    # echo_pin2 = 23
    # trigger_pin2 = 24
    
    # start_thread(sample_rate, echo_pin2, trigger_pin2)
    try:
        while True:
            time.sleep(1)
            print(time.time())
            # Safely copy the shared target dictionary
            with ULTRASONIC_INFO_LOCK:
                ultrasonic_info = dict(GLOBAL_ULTRASONIC_INFO)

            for name, info in ultrasonic_info.items():
                print(f"{name}: "
                        f"distance ={info['US_distance_cm']:.2f} "
                        f"={info['ts']:.2f} "
                )

            if(time.time() > start_time + 60):
                print(" --------------------------------------------- STOP THREAD -------------------------")
                stop_thread()
                break

    except: 
        print("exception a thrown")


