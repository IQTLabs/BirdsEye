"""
Tests for sigscan.py
"""
from sigscan import SigScan


class MockMessageObject:
    """
    Mock class for MQTT message objects
    """

    def __init__(self, payload):
        """
        Initialize variables
        """
        self.payload = payload


def test_sigscan():
    """
    Test the SigScan class
    """
    instance = SigScan(config_path="tests/test_sigscan_config.ini")
    instance.main()
    instance = SigScan(config_path="tests/test_sigscan_config2.ini")
    instance.main()
    instance = SigScan(config_path="tests/test_sigscan_config3.ini")
    instance.main()
    instance = SigScan(config_path="tests/test_sigscan_config4.ini")
    instance.main()


def test_on_message():
    """
    Test MQTT messages
    """
    instance = SigScan(config_path="tests/test_sigscan_config.ini")
    messages = []
    with open("tests/mqtt_messages.log", "r", encoding="UTF-8") as file:
        for line in file:
            messages.append(line.strip())
    for message in messages:
        msg_obj = MockMessageObject(message)
        instance.on_message(None, None, msg_obj)
