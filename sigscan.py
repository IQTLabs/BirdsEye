import base64
import configparser
import json
import logging
import sys
import threading
from datetime import datetime
from io import BytesIO
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import paho.mqtt.client as mqtt
import torch
from flask import Flask

import birdseye.dqn
import birdseye.env
import birdseye.mcts_utils
import birdseye.sensor
import birdseye.state
import birdseye.utils
from birdseye.actions import WalkingActions
from birdseye.planner import DQNPlanner
from birdseye.planner import MCTSPlanner
from birdseye.utils import get_heading
from birdseye.utils import get_distance
from birdseye.utils import is_float

logging.basicConfig(level=10, format="%(asctime)s %(message)s")
logging.getLogger("matplotlib.font_manager").disabled = True


class GamutRFSensor(birdseye.sensor.SingleRSSI):
    """
    GamutRF Sensor
    """

    def __init__(
        self,
        antenna_filename=None,
        power_tx=26,
        directivity_tx=1,
        freq=5.7e9,
        fading_sigma=None,
        threshold=-120,
        data={},
    ):
        super().__init__(
            antenna_filename=antenna_filename,
            power_tx=power_tx,
            directivity_tx=directivity_tx,
            freq=freq,
            fading_sigma=fading_sigma,
        )
        self.threshold = threshold
        self.data = data

    def real_observation(self):
        if (self.data.get("rssi", None)) is None or (
            self.data["rssi"] < self.threshold
        ):
            return None
        return self.data["rssi"]


class SigScan:
    def __init__(self, config_path="sigscan_config.ini"):
        self.data = {
            "rssi": None,
            "position": None,
            "distance": None,
            "previous_position": None,
            "heading": None,
            "previous_heading": None,
            "course": None,
            "action_proposal": None,
            "action_taken": None,
            "reward": None,
        }
        config = configparser.ConfigParser()
        config.read(config_path)
        self.config = config["sigscan"]
        self.static_position = None
        self.static_heading = None

    def data_handler(self, message_data):
        """
        Generic data processor
        """
        if self.static_position:
            message_data["position"] = self.static_position
        if self.static_heading is not None:
            message_data["heading"] = self.static_heading

        self.data["previous_position"] = (
            self.data.get("position", None)
            if not self.data.get("needs_processing", True)
            else self.data.get("previous_position", None)
        )
        self.data["previous_heading"] = (
            self.data.get("heading", None)
            if not self.data.get("needs_processing", True)
            else self.data.get("previous_heading", None)
        )

        self.data["rssi"] = message_data.get("rssi", None)
        self.data["position"] = message_data.get("position", None)
        self.data["course"] = get_heading(
            self.data["previous_position"], self.data["position"]
        )
        self.data["heading"] = (
            -float(message_data.get("heading", None)) + 90
            if is_float(message_data.get("heading", None))
            else self.data["course"]
        )
        self.data["distance"] = get_distance(
            self.data["previous_position"], self.data["position"]
        )
        delta_heading = (
            (self.data["heading"] - self.data["previous_heading"])
            if self.data["heading"] and self.data["previous_heading"]
            else None
        )
        self.data["action_taken"] = (
            (delta_heading, self.data["distance"])
            if delta_heading and self.data["distance"]
            else (0, 0)
        )

        self.data["drone_position"] = message_data.get("drone_position", None)
        if self.data["drone_position"]:
            self.data["drone_position"] = [
                self.data["drone_position"][1],
                self.data["drone_position"][0],
            ]  # swap lon,lat

        self.data["needs_processing"] = True

    def on_message(self, client, userdata, json_message):
        """
        Get MQTT messages
        """
        json_data = json.loads(json_message.payload)
        self.data_handler(json_data)

    def on_connect(self, client, userdata, flags, result_code):
        """
        Subscribe to MQTT channel
        """
        sub_channel = "gamutrf/rssi"
        logging.info(
            "Connected to %s with result code %s", sub_channel, str(result_code)
        )
        client.subscribe(sub_channel)

    def run_flask(self, flask_host, flask_port, fig, results):
        """
        Flask
        """
        app = Flask(__name__)

        @app.route("/")
        def hello():
            # Save figure to a temporary buffer.
            flask_start_time = timer()
            buf = BytesIO()

            try:
                fig.savefig(buf, format="png", bbox_inches="tight")
            except ValueError:
                return '<html><head><meta http-equiv="refresh" content="1"></head><body><p>No image, refreshing...</p></body></html>'

            # Embed the result in the html output.
            data = base64.b64encode(buf.getvalue()).decode("ascii")
            flask_end_time = timer()

            logging.debug("=======================================")
            logging.debug("Flask Timing")
            logging.debug("time step = %s", str(results.time_step))
            logging.debug("buffer size = {:.2f} MB".format(len(buf.getbuffer()) / 1e6))
            logging.debug(
                "Duration = {:.4f} s".format(flask_end_time - flask_start_time)
            )
            logging.debug("=======================================")
            return f'<html><head><meta http-equiv="refresh" content="0.5"></head><body><img src="data:image/png;base64,{data}"/></body></html>'

        host_name = flask_host
        port = flask_port
        threading.Thread(
            target=lambda: app.run(
                host=host_name, port=port, debug=False, use_reloader=False
            )
        ).start()

    def main(self):
        """
        Main loop
        """
        static_position = self.config.get("static_position", None)
        if static_position:
            static_position = [float(i) for i in static_position.split(",")]
        self.static_position = static_position

        static_heading = self.config.get("static_heading", None)
        if static_heading is not None:
            static_heading = float(static_heading)
        self.static_heading = static_heading

        replay_file = self.config.get("replay_file", None)

        mqtt_host = self.config.get("mqtt_host", "localhost")
        mqtt_port = int(self.config.get("mqtt_port", str(1883)))

        flask_host = self.config.get("flask_host", "127.0.0.1")
        flask_port = int(self.config.get("flask_port", str(4999)))

        n_antennas = int(self.config.get("n_antennas", str(1)))
        antenna_type = self.config.get("antenna_type", "omni")
        planner_method = self.config.get("planner_method", "dqn")
        power_tx = float(self.config.get("power_tx", str(26)))
        directivity_tx = float(self.config.get("directivity_tx", str(1)))
        freq = float(self.config.get("freq", str(5.7e9)))
        fading_sigma = float(self.config.get("fading_sigma", str(8)))
        threshold = float(self.config.get("threshold", str(-120)))
        reward_func = self.config.get("reward", "heuristic_reward")
        n_targets = int(self.config.get("n_targets", str(2)))
        dqn_checkpoint = self.config.get("dqn_checkpoint", None)
        if planner_method in ["dqn", "DQN"] and dqn_checkpoint is None:
            if n_antennas == 1 and antenna_type == "directional" and n_targets == 2:
                dqn_checkpoint = (
                    "checkpoints/single_directional_entropy_walking.checkpoint"
                )
            elif n_antennas == 1 and antenna_type == "omni":
                dqn_checkpoint = "checkpoints/single_omni_entropy_walking.checkpoint"
            elif n_antennas == 2 and antenna_type == "directional" and n_targets == 2:
                dqn_checkpoint = (
                    "checkpoints/double_directional_entropy_walking.checkpoint"
                )
            elif n_antennas == 2 and antenna_type == "directional" and n_targets == 1:
                dqn_checkpoint = (
                    "checkpoints/double_directional_entropy_walking_1target.checkpoint"
                )
            elif n_antennas == 1 and antenna_type == "directional" and n_targets == 1:
                dqn_checkpoint = (
                    "checkpoints/single_directional_entropy_walking_1target.checkpoint"
                )

        # MQTT
        if replay_file is None:
            try:
                client = mqtt.Client()
                client.on_connect = self.on_connect
                client.on_message = self.on_message
                client.connect(mqtt_host, mqtt_port, 60)
                client.loop_start()
            except Exception as err:
                logging.error(
                    "Unable to connect to MQTT host %s:%s because: %s.",
                    mqtt_host,
                    str(mqtt_port),
                    str(err),
                )
                sys.exit(1)
        else:
            with open(replay_file, "r", encoding="UTF-8") as open_file:
                replay_data = json.load(open_file)
                replay_ts = sorted(replay_data.keys())

        # BirdsEye
        global_start_time = datetime.utcnow().timestamp()
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # pylint: disable=no-member
        results = birdseye.utils.Results(
            method_name=planner_method,
            global_start_time=global_start_time,
            config=self.config,
        )

        # Sensor
        if antenna_type in ["directional", "yagi", "logp"]:
            antenna_filename = "radiation_pattern_yagi_5.csv"
        elif antenna_type in ["omni", "omnidirectional"]:
            antenna_filename = "radiation_pattern_monopole.csv"

        sensor = GamutRFSensor(
            antenna_filename=antenna_filename,
            power_tx=power_tx,
            directivity_tx=directivity_tx,
            freq=freq,
            fading_sigma=fading_sigma,
            threshold=threshold,
            data=self.data,
        )  # fading sigm = 8dB, threshold = -120dB

        # Action space
        actions = WalkingActions()
        actions.print_action_info()

        # State managment
        state = birdseye.state.RFMultiState(
            n_targets=n_targets, reward=reward_func, simulated=False
        )

        # Environment
        env = birdseye.env.RFMultiEnv(
            sensor=sensor, actions=actions, state=state, simulated=False
        )
        belief = env.reset()

        # Motion planner
        if self.config.get("use_planner", "false").lower() != "true":
            planner = None
        elif planner_method in ["dqn", "DQN"]:
            planner = DQNPlanner(env, actions, device, dqn_checkpoint)
        elif planner_method in ["mcts", "MCTS"]:
            depth = 2
            c = 20
            simulations = 50
            planner = MCTSPlanner(env, actions, depth, c, simulations)
        else:
            raise ValueError("planner_method not valid")

        # Flask
        fig = plt.figure(figsize=(18, 10), dpi=50)
        ax = fig.subplots()
        fig.set_tight_layout(True)
        time_step = 0
        if self.config.get("flask", "false").lower() == "true":
            self.run_flask(flask_host, flask_port, fig, results)

        # Main loop
        while True:
            loop_start = timer()
            self.data["utc_time"] = datetime.utcnow().timestamp()
            time_step += 1

            if replay_file is not None:
                # load data from saved file
                if time_step - 1 == len(replay_ts):
                    break
                self.data_handler(replay_data[replay_ts[time_step - 1]])

            action_start = timer()
            self.data["action_proposal"] = (
                planner.proposal(belief) if planner else [None, None]
            )
            action_end = timer()

            step_start = timer()
            # update belief based on action and sensor observation (sensor is read inside)
            if self.data.get("needs_processing", False):
                belief, reward, observation = env.real_step(self.data)
                self.data["reward"] = reward
                self.data["needs_processing"] = False
            step_end = timer()

            plot_start = timer()
            results.live_plot(
                env=env, time_step=time_step, fig=fig, ax=ax, data=self.data
            )
            plot_end = timer()

            particle_save_start = timer()
            np.save(
                f'{results.logdir}/{self.data["utc_time"]}_particles.npy',
                env.pf.particles,
            )
            particle_save_end = timer()

            data_start = timer()
            with open(
                f"{results.logdir}/birdseye-{global_start_time}.log",
                "a",
                encoding="UTF-8",
            ) as outfile:
                json.dump(self.data, outfile)
                outfile.write("\n")
            data_end = timer()

            loop_end = timer()

            logging.debug("=======================================")
            logging.debug("BirdsEye Timing")
            logging.debug("time step = {}".format(time_step))
            logging.debug(
                "action selection = {:.4f} s".format(action_end - action_start)
            )
            logging.debug("env step = {:.4f} s".format(step_end - step_start))
            logging.debug("plot = {:.4f} s".format(plot_end - plot_start))
            logging.debug(
                "particle save = {:.4f} s".format(
                    particle_save_end - particle_save_start
                )
            )
            logging.debug("data save = {:.4f} s".format(data_end - data_start))
            logging.debug("main loop = {:.4f} s".format(loop_end - loop_start))
            logging.debug("=======================================")

        if self.config.get("make_gif", "false").lower() == "true":
            results.save_gif("tracking")


if __name__ == "__main__":  # pragma: no cover
    instance = SigScan()
    instance.main()
