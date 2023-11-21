import json
import logging
import paho.mqtt.client
import sys


class BirdsEyeMQTT:
    def __init__(self, mqtt_host, mqtt_port, message_handler):
        try:
            self.client = paho.mqtt.client.Client()
            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            self.client.on_publish = self.on_publish
            self.message_handler = message_handler
            self.client.connect(mqtt_host, mqtt_port, 60)
            self.client.loop_start()

        except Exception as err:
            logging.error(
                "Unable to connect to MQTT host %s:%s because: %s.",
                mqtt_host,
                str(mqtt_port),
                str(err),
            )
            sys.exit(1)

    def on_message(self, client, userdata, json_message):
        json_data = json.loads(json_message.payload)
        self.message_handler(json_data)

    def on_connect(self, client, userdata, flags, result_code):
        sub_channel = "gamutrf/rssi"
        logging.info(
            "Connected to %s with result code %s", sub_channel, str(result_code)
        )
        self.client.subscribe(sub_channel)  # also tried qos = 1 and 1

    def on_publish(self, client, userdata, mid):
        logging.info("Completed transmission to broker.")
