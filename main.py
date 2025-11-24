# main.py

import time
# /home/clopezgarcia2/Desktop/roboforge/roboforge_vision/traingulate_w_yolo.py
from roboforge_vision.traingulate_w_yolo import (
    start_thread,
    stop_thread,
    GLOBAL_TARGET_INFO,
    TARGET_INFO_LOCK
)


def main():
    print("[Main] Starting stereo vision thread...")
    start_thread()

    try:
        while True:
            time.sleep(0.1)

            # Safely copy the shared target dictionary
            with TARGET_INFO_LOCK:
                targets = dict(GLOBAL_TARGET_INFO)

            if targets:
                print("\n=== TARGETS FROM VISION THREAD ===")
                for name, info in targets.items():
                    print(f"{name}: "
                          f"X={info['X']:.2f} "
                          f"Y={info['Y']:.2f} "
                          f"Z={info['Z']:.2f} "
                          f"D={info['D']:.2f} "
                          f"Conf={info['confidence']:.2f}")

            # YOUR ARM / SERVO / ROBOT DECISIONS GO HERE
            # Example:
            # if 'cube' in targets:
            #     print("Cube found, moving arm...")

    except KeyboardInterrupt:
        print("\n[Main] Keyboard interrupt — shutting down.")

    finally:
        stop_thread()
        time.sleep(0.5)
        print("[Main] Done.")


if __name__ == "__main__":
    main()
