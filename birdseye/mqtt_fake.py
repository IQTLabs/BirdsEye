import argparse
import json
import logging

from pynput import keyboard
from pynput.keyboard import Key

from birdseye.mqtt import BirdsEyeMQTT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="INFO")
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % args.log)
    logging.basicConfig(level=numeric_level)

    def message_handler(data):
        logging.info("Message handler received data: {}".format(data))

    mqtt_client = BirdsEyeMQTT("localhost", 1883, message_handler)

    publish_data = {
        "position": [32.922651, -117.120815],
        "heading": 0,
        "rssi": -40,
    }

    def on_key_release(key):
        if key == Key.right:
            # print("Right key clicked")
            publish_data["position"][1] += 0.0001
        elif key == Key.left:
            # print("Left key clicked")
            publish_data["position"][1] -= 0.0001
        elif key == Key.up:
            # print("Up key clicked")
            publish_data["position"][0] += 0.0001
        elif key == Key.down:
            # print("Down key clicked")
            publish_data["position"][0] -= 0.0001
        elif key == Key.esc:
            exit()

        mqtt_client.client.publish(
            "gamutrf/rssi", json.dumps(publish_data)
        )  # also tried qos = 1 and 2
        logging.info("Started transmission to broker: {}".format(publish_data))

    with keyboard.Listener(on_release=on_key_release) as listener:
        listener.join()
