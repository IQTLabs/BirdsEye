import json
import logging
import paho.mqtt.client
import sys


class BirdsEyeMQTT:
    def __init__(self, mqtt_host, mqtt_port, topics):
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.topics = topics
        try:
            self.client = paho.mqtt.client.Client()
            self.client.on_connect = self.on_connect
            #self.client.on_message = self.on_message
            self.client.on_publish = self.on_publish
            #self.message_handler = message_handler
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

    def on_message_func(self, message_handler):
        def on_message(client, userdata, json_message):
            json_data = json.loads(json_message.payload)
            message_handler(json_data)
        return on_message

    def on_connect(self, client, userdata, flags, result_code):
        logging.info(
            f"Connected to MQTT broker {self.mqtt_host}:{self.mqtt_port} with result code {result_code}"
        )
        for topic, handler in self.topics:
            self.client.subscribe(topic) 
            self.client.message_callback_add(topic, self.on_message_func(handler))

    def on_publish(self, client, userdata, mid):
        logging.info("Completed transmission to broker.")
