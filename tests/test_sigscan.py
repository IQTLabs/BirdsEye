"""
Tests for sigscan.py
"""
# import signal

# import httpx
# import matplotlib.pyplot as plt
# import pytest

# from birdseye.utils import Results
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


# def test_run_flask():
#    """
#    Test the run_flask function
#    """
#    instance = SigScan(config_path="tests/test_sigscan_config.ini")
#    results  = Results(
#        method_name="dqn", global_start_time="1656695290", plotting="True"
#    )
#    fig = plt.figure(figsize=(18, 10), dpi=50)
#    ax = fig.subplots()
#    fig.set_tight_layout(True)
#    instance.run_flask('127.0.0.1', 1111, fig, results)
#    request = httpx.get('http://127.0.0.1:1111/')
#    assert request.status_code == 200
#    with pytest.raises(KeyboardInterrupt) as pytest_wrapped_e:
#        signal.raise_signal(signal.SIGINT)
#    assert pytest_wrapped_e.type == KeyboardInterrupt


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
