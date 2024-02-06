import argparse
import json
import logging

from pynput import keyboard
from pynput.keyboard import Key

from birdseye.mqtt import BirdsEyeMQTT


def main(blocking=True):
    def message_handler(data):
        logging.info("Message handler received data: {}".format(data))

    mqtt_client = BirdsEyeMQTT(
        "localhost",
        1883,
        [("gamutrf/inference", message_handler), ("gamutrf/targets", message_handler)],
    )

    sensor_data = {
        "metadata": {
            "image_path": "/logs/inference/image_1700632876.885_640x640_2408895999Hz.png",
            "orig_rows": "28897",
            "rssi_max": "-46.058800",
            "rssi_mean": "-69.604271",
            "rssi_min": "-117.403526",
            "rx_freq": "2408895999",
            "ts": "1700632876.885",
        },
        "predictions": {
            "mini2_telem": [
                {
                    "rssi": "-40",
                    "conf": "0.33034399151802063",
                    "xywh": [
                        609.8685302734375,
                        250.76278686523438,
                        20.482666015625,
                        7.45684814453125,
                    ],
                }
            ],
            "mini2_video": [
                {
                    "rssi": "-80",
                    "conf": "0.33034399151802063",
                    "xywh": [
                        609.8685302734375,
                        250.76278686523438,
                        20.482666015625,
                        7.45684814453125,
                    ],
                }
            ],
        },
        "position": [32.922651, -117.120815],
        "heading": 0,
        "rssi": [-40, -60],
        "gps": "fix",
    }
    target_data = {
        "altitude": 4700,
        "gps_fix_type": 2,
        "gps_stale": "false",
        "heading": 174.15,
        "latitude": 32.922651,
        "longitude": -117.120815,
        "relative_alt": 4703,
        "target_name": "drone1",
        "time_boot_ms": 904414,
        "time_usec": None,
        "vx": 0.0,
        "vy": 0.0,
        "vz": 0.0,
    }

    control_options = {"sensor": sensor_data, "target": target_data}
    control_map = {i: k for (i, k) in enumerate(control_options)}
    control_key = None
    help_str = f"\nChoose the device to control by entering a number from {list(range(len(control_options)))}.\n{control_map}\n"
    print(help_str)

    def on_key_release(key):
        nonlocal control_key

        if key == Key.esc:
            exit()

        try:
            if key.char == "0":
                print(f"\nSelected sensor. Up/Down/Left/Right will control sensor.")
                control_key = "sensor"
            elif key.char == "1":
                print(f"\nSelected target 1. Up/Down/Left/Right will control target 1.")
                control_key = "target"
            
        except:
            pass

        if control_key is None:
            print(help_str)
            return

        try:

            if key.char == "n":
                print(f"\n\nn key pressed. Sending no gps fix.")
                if control_key == "sensor":
                    sensor_data["gps"] = "no_fix"
                elif control_key == "target":
                    target_data["gps_fix_type"] = "null"
            elif key.char == "g":
                print(f"\n\ng key pressed. Sending good gps fix.")
                if control_key == "sensor":
                    sensor_data["gps"] = "fix"
                elif control_key == "target":
                    target_data["gps_fix_type"] = 2 
            return
        except:
            pass

        if key == Key.right:
            print("\n\nRight key pressed. Moving right.\n")
            if control_key == "sensor":
                sensor_data["position"][1] += 0.0001
            elif control_key == "target":
                target_data["longitude"] += 0.0001
        elif key == Key.left:
            print("\n\nLeft key pressed. Moving left.\n")
            if control_key == "sensor":
                sensor_data["position"][1] -= 0.0001
            elif control_key == "target":
                target_data["longitude"] -= 0.0001
        elif key == Key.up:
            print("\n\nUp key pressed. Moving up.\n")
            if control_key == "sensor":
                sensor_data["position"][0] += 0.0001
            elif control_key == "target":
                target_data["latitude"] += 0.0001
        elif key == Key.down:
            print("\n\nDown key pressed. Moving down.\n")
            if control_key == "sensor":
                sensor_data["position"][0] -= 0.0001
            elif control_key == "target":
                target_data["latitude"] -= 0.0001
        else:
            print(f"Press Up/Down/Left/Right to control {control_key}.\n")
            print(f"To change device being controlled:\n {help_str}")
            return
          

        

        if control_key == "sensor":
            mqtt_client.client.publish(
                "gamutrf/inference", json.dumps(sensor_data)
            )  # also tried qos = 1 and 2
            logging.info("Started transmission to broker: {}".format(sensor_data))
        elif control_key == "target":
            mqtt_client.client.publish("gamutrf/targets", json.dumps(target_data))
            logging.info("Started transmission to broker: {}".format(target_data))

    if blocking:
        with keyboard.Listener(on_release=on_key_release) as listener:
            listener.join()
    else:
        listener = keyboard.Listener(on_release=on_key_release)
        listener.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="INFO", help="Log level.")
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % args.log)
    logging.basicConfig(level=numeric_level)
    main()
