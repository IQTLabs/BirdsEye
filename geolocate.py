import argparse
import base64
import configparser
import json
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import threading
import time

from datetime import datetime
from flask import (
    Flask,
    render_template,
    send_from_directory,
    redirect,
    request,
    url_for,
    make_response,
    jsonify,
)
from io import BytesIO
from timeit import default_timer as timer

import birdseye.env
import birdseye.mqtt
import birdseye.sensor
import birdseye.state
import birdseye.utils
from birdseye.planners.light_mcts import LightMCTS
from birdseye.planners.lavapilot import LAVAPilot
from birdseye.planners.repp import REPP
from birdseye.utils import (
    get_heading,
    get_distance,
    is_float,
    tracking_metrics_separable,
    targets_found,
)

ORCHESTRATOR = os.getenv("ORCHESTRATOR", "0.0.0.0")  # nosec


class GamutRFSensor(birdseye.sensor.SingleRSSISeparable):
    """
    GamutRF Sensor
    """

    def __init__(
        self,
        antenna_filename=None,
        power_tx=26,
        directivity_tx=1,
        freq=5.7e9,
        n_targets=1,
        fading_sigma=None,
        threshold=-120,
        data={},
    ):
        super().__init__(
            antenna_filename=antenna_filename,
            power_tx=power_tx,
            directivity_tx=directivity_tx,
            freq=freq,
            n_targets=n_targets,
            fading_sigma=fading_sigma,
        )
        self.threshold = threshold
        self.data = data
        self.n_targets = n_targets
        self.class_map = {}

    def real_observation(self):
        observation = self.data.get("rssi", None)
        default_observation = [None] * self.n_targets

        if observation is None:
            return default_observation

        if type(observation) == dict:
            format_observation = default_observation
            for class_name in observation:
                self.class_map[class_name] = self.class_map.get(
                    class_name, len(self.class_map)
                )
                format_observation[self.class_map[class_name]] = observation[class_name]
            observation = format_observation

        if type(observation) != list:
            observation = [observation]

        if len(observation) != self.n_targets:
            raise ValueError("len(observation) != n_targets")

        for i in range(len(observation)):
            if observation[i] and observation[i] < self.threshold:
                observation[i] = None

        return observation


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Geolocate:
    def __init__(self, config_path="geolocate.ini"):
        self.init_data()
        config = configparser.ConfigParser()
        config.read(config_path)
        self.config = config["geolocate"]
        self.config_path = config_path
        self.static_position = None
        self.static_heading = None
        self.setDaemon = False

        #### CONFIGS
        default_config = {
            "local_plot": "false",
            "make_gif": "false",
            "n_targets": "2",
            "antenna_type": "logp",
            "planner_method": "repp",
            "target_speed": "0.5",
            "sensor_speed": "1.0",
            "power_tx": "26.0",
            "directivity_tx": "1.0",
            "freq": "5.7e9",
            "fading_sigma": "8.0",
            "threshold": "-120",
            "mcts_depth": "3",
            "mcts_c": "20.0",
            "mcts_simulations": "100",
            "mcts_n_downsample": "400",
            "static_position": None,
            "static_heading": None,
            "replay_file": None,
            "mqtt_host": ORCHESTRATOR,
            "mqtt_port": "1883",
            "flask_host": "0.0.0.0",  # nosec
            "flask_port": "4999",
            "use_flask": "false",
        }
        default_config.update(self.config)
        self.config = default_config

    def init_data(
        self,
    ):
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
            "needs_processing": False,
            "gps": None,
            "targets": {},
            "target_gps": None,
        }

    def target_handler(self, message_data):
        logging.info(f"Received gamutrf/target MQTT message: {message_data}")

        if (
            message_data["gps_stale"].lower() in ["null", "true"]
            or (
                isinstance(message_data["gps_fix_type"], str)
                and message_data["gps_fix_type"].lower() == "null"
            )
            or (
                isinstance(message_data["gps_fix_type"], (int, float))
                and int(message_data["gps_fix_type"]) < 2
            )
        ):
            logging.info(f"No target GPS")
            return

        self.data["target_gps"] = "fix"
        target_name = message_data["target_name"]
        if target_name not in self.data["targets"]:
            self.data["targets"][target_name] = {"idx": len(self.data["targets"])}

        self.data["targets"][target_name]["position"] = (
            message_data["latitude"],
            message_data["longitude"],
        )

    def data_handler(self, message_data):
        """
        Generic data processor
        """

        logging.info(f"Received MQTT message: {message_data}")
        if self.data["needs_processing"]:
            logging.debug("\nReceived multiple data in one step!\n")
        if self.static_position:
            message_data["position"] = self.static_position
        if self.static_heading is not None:
            message_data["heading"] = self.static_heading

        self.data["previous_position"] = (
            self.data["position"]
            if not self.data["needs_processing"]
            else self.data["previous_position"]
        )
        self.data["previous_heading"] = (
            self.data["heading"]
            if not self.data["needs_processing"]
            else self.data["previous_heading"]
        )

        self.data["gps"] = message_data.get("gps", None)

        predictions = message_data.get("predictions", None)
        metadata = message_data.get("metadata", None)
        if predictions:
            self.data["rssi"] = {}
            for class_name in predictions.keys():
                self.data["rssi"][class_name] = np.mean(
                    [
                        float(prediction.get("rssi_max", metadata["rssi_max"]))
                        for prediction in predictions[class_name]
                    ]
                )

        elif metadata:
            # TODO: how to handle when tracking multiple targets
            # self.data["rssi"] = float(metadata["rssi_mean"])
            # self.data["rssi"] = float(metadata["rssi_min"])
            # self.data["rssi"] = float(metadata.get["rssi_max"])
            pass
        else:
            self.data["rssi"] = message_data.get("rssi", None)

        self.data["position"] = message_data.get("position", self.data["position"])

        # course is direction of movement
        self.data["course"] = get_heading(
            self.data["previous_position"], self.data["position"]
        )

        # heading is antenna facing direction
        # mavlink heading is yaw relative to North
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

    def run_flask(self, flask_host, flask_port, fig, results):
        """
        Flask
        """
        app = Flask(__name__)

        @app.route("/gui/<path:filename>")
        def gui_file(filename):
            return send_from_directory("gui", filename)

        @app.route("/gui/data")
        def gui_data():
            data = base64.b64encode(self.image_buf.getvalue()).decode("ascii")
            img = "data:image/png;base64," + data
            return jsonify(img)

        @app.route("/refresh")
        def refresh():
            os.makedirs("gui", exist_ok=True)
            with open("gui/map.png", "wb") as img:
                img.write(self.image_buf.getbuffer())
            # newmapp = np.random.rand(500,500,3) * 255
            # data = Image.fromarray(newmapp.astype('uint8')).convert('RGBA')
            # data.save('gui/map.png')
            return "OK"

        @app.route("/gui/form", methods=["POST", "GET"])
        def gui_form():
            if request.method == "POST":
                user = request.form.get("name", None)
                self.config["n_targets"] = request.form.get(
                    "n_targets", self.config["n_targets"]
                )
                reset = request.form.get("reset", None)
                if reset == "reset":
                    self.stop()
                    self.image_buf = BytesIO()
                    self.start()
                return redirect(request.referrer)

        @app.route("/guiFile")
        def gui_from_file():
            if not self.image_buf.getbuffer().nbytes:
                return render_template("loading.html")
            return render_template("gui_from_file.html", config=self.config)

        @app.route("/guiBuffer")
        def gui_from_buffer():
            if not self.image_buf.getbuffer().nbytes:
                return render_template("loading.html")
            return render_template("gui_from_buffer.html", config=self.config)

        @app.route("/")
        def index():
            flask_start_time = timer()

            if not self.image_buf.getbuffer().nbytes:
                return render_template("loading.html")

            data = base64.b64encode(self.image_buf.getvalue()).decode("ascii")

            flask_end_time = timer()

            logging.debug("=======================================")
            logging.debug("Flask Timing")
            logging.debug("time step = %s", str(results.time_step))
            logging.debug(
                "buffer size = {:.2f} MB".format(len(self.image_buf.getbuffer()) / 1e6)
            )
            logging.debug(
                "Duration = {:.4f} s".format(flask_end_time - flask_start_time)
            )
            logging.debug("=======================================")

            return render_template("birdseye_live.html", data=data)

        host_name = flask_host
        port = flask_port
        self.flask_thread = threading.Thread(
            target=lambda: app.run(
                host=host_name, port=port, debug=False, use_reloader=False
            )
        )
        self.flask_thread.daemon = self.setDaemon
        self.flask_thread.start()

    def get_replay_json(self, replay_file):
        with open(replay_file, "r", encoding="UTF-8") as open_file:
            replay_data = json.load(open_file)
            for ts in replay_data:
                yield replay_data[ts]

    def get_replay_log(self, replay_file):
        with open(replay_file, "r", encoding="UTF-8") as open_file:
            for line in open_file:
                replay_data = json.loads(line)
                yield replay_data

    def start(self, setDaemon=False):
        self.setDaemon = setDaemon
        self.stop_threads = False
        self.init_data()
        self.main_thread = threading.Thread(
            target=self.main, args=[lambda: self.stop_threads]
        )
        self.main_thread.start()
        logging.info("Main thread started.")

    def stop(self):
        self.stop_threads = True
        logging.info("Main thread stopped.")
        self.main_thread.join()

    def main(self, stopped):
        """
        Main loop
        """
        self.static_position = self.config["static_position"]
        if self.static_position:
            self.static_position = [float(i) for i in self.static_position.split(",")]
            self.data["position"] = self.static_position

        self.static_heading = self.config["static_heading"]
        if self.static_heading:
            self.static_heading = float(self.static_heading)
            self.data["heading"] = self.static_heading

        replay_file = self.config["replay_file"]

        mqtt_host = self.config["mqtt_host"]
        mqtt_port = int(self.config["mqtt_port"])

        flask_host = self.config["flask_host"]
        flask_port = int(self.config["flask_port"])

        antenna_type = self.config["antenna_type"]
        planner_method = self.config["planner_method"]

        n_targets = int(self.config["n_targets"])

        sensor_speed = float(self.config["sensor_speed"])
        target_speed = float(self.config["target_speed"])

        power_tx = self.config["power_tx"]
        power_tx = [float(x) for x in power_tx.split(",")]
        if len(power_tx) == 1:
            power_tx = [power_tx[0] for _ in range(n_targets)]

        directivity_tx = self.config["directivity_tx"]
        directivity_tx = [float(x) for x in directivity_tx.split(",")]
        if len(directivity_tx) == 1:
            directivity_tx = [directivity_tx[0] for _ in range(n_targets)]

        freq = self.config["freq"]
        freq = [float(x) for x in freq.split(",")]
        if len(freq) == 1:
            freq = [freq[0] for _ in range(n_targets)]

        fading_sigma = float(self.config["fading_sigma"])
        threshold = float(self.config["threshold"])

        particle_distance = float(self.config["particle_distance"])

        local_plot = self.config["local_plot"].lower()
        make_gif = self.config["make_gif"].lower()
        use_flask = self.config["use_flask"].lower()
        if (local_plot == "true") or (make_gif == "true") or (use_flask == "true"):
            any_plot = True
        else:
            any_plot = False
        ##########

        ###### MQTT or replay from file
        if replay_file is None:
            topics = [
                ("gamutrf/inference", self.data_handler),
                ("gamutrf/targets", self.target_handler),
            ]
            mqtt_client = birdseye.mqtt.BirdsEyeMQTT(mqtt_host, mqtt_port, topics)
        else:
            if replay_file.endswith(".log"):
                get_replay_data = self.get_replay_log(replay_file)
            elif replay_file.endswith(".json"):
                get_replay_data = self.get_replay_json(replay_file)
        ###########

        # BirdsEye
        global_start_time = datetime.utcnow().timestamp()
        n_simulations = 100
        max_iterations = 400
        reward_func = (
            lambda pf, **kwargs: pf.weight_entropy
        )  # lambda *args, **kwargs: None
        r_min = 10
        horizon = 1  # 8
        min_bound = 0.82
        min_std_dev = 35
        num_particles = 3000  # 3000
        step_duration = 1

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
            n_targets=n_targets,
            fading_sigma=fading_sigma,
            threshold=threshold,
            data=self.data,
        )  # fading sigm = 8dB, threshold = -120dB

        # Test expected RSSI
        # for distance in [10, 20, 30, 40, 50, 60, 70]:
        #     print(f"{distance=}, rssi={sensor.observation([distance,0],0,0)}")

        # Action space
        # actions = WalkingActions()
        actions = birdseye.actions.BaselineActions(sensor_speed=sensor_speed)
        actions.print_action_info()

        # State managment
        state = birdseye.state.RFMultiState(
            n_targets=n_targets,
            target_speed=target_speed,
            sensor_speed=sensor_speed,
            reward=reward_func,
            simulated=False,
        )

        # Environment
        env = birdseye.env.RFMultiSeparableEnv(
            sensor=sensor,
            actions=actions,
            state=state,
            simulated=False,
            num_particles=num_particles,
        )

        belief = env.reset()

        results = birdseye.utils.Results(
            experiment_name=self.config_path,
            global_start_time=global_start_time,
            config=self.config,
            class_map=sensor.class_map,
        )

        # Motion planner
        if self.config.get("use_planner", "false").lower() != "true":
            planner = None
        else:
            target_selections = {t for t in range(n_targets)}
            if planner_method == "repp":  # REPP
                planner = REPP(
                    env, min_std_dev, r_min, horizon, min_bound, target_selections
                )
            elif planner_method == "lavapilot":  # LAVAPilot
                planner = LAVAPilot(env, min_std_dev, r_min, horizon, min_bound)
            elif planner_method == "mcts":  # MCTS
                mcts_depth = int(self.config["mcts_depth"])
                mcts_c = float(self.config["mcts_c"])
                mcts_simulations = int(self.config["mcts_simulations"])
                mcts_n_downsample = int(self.config["mcts_n_downsample"])
                planner = LightMCTS(
                    env,
                    depth=mcts_depth,
                    c=mcts_c,
                    simulations=mcts_simulations,
                    n_downsample=mcts_n_downsample,
                )
            else:
                raise Exception

        if use_flask == "true":
            matplotlib.use("agg")
        if any_plot:
            fig = plt.figure(figsize=(14, 10), dpi=100)
            ax = fig.subplots()
            fig.set_layout_engine("tight")
            plt.show(block=False)

        self.image_buf = BytesIO()
        if use_flask == "true":
            self.run_flask(flask_host, flask_port, fig, results)

        ##############
        # Main loop
        ##############
        time_step = 0
        control_actions = []
        step_time = 0

        while (
            self.data["gps"] != "fix"
            and self.data["target_gps"] != "fix"
            and not replay_file
            and not stopped()
        ):
            time.sleep(1)
            logging.info("Waiting for GPS...")

        while True and not stopped():
            loop_start = timer()
            self.data["utc_time"] = datetime.utcnow().timestamp()

            if replay_file:
                # load data from saved file
                try:
                    replay_data = next(get_replay_data)
                except StopIteration:
                    break

                self.data_handler(replay_data)

            action_start = timer()

            if planner:
                if time_step % horizon == 0:
                    if targets_found(env, min_std_dev):
                        # all objects localized
                        control_action = [None]

                    else:
                        plan_start_time = timer()
                        control_action = planner.get_action()
                        plan_end_time = timer()

                    control_actions.extend(control_action)
                # logging.info(f"{control_actions[-1]=}")
                action = control_actions[time_step]

                self.data["action_proposal"] = action

            action_end = timer()

            step_start = timer()

            while time.perf_counter() - step_time < step_duration:
                pass
            step_time = time.perf_counter()
            observation = env.real_step(self.data)
            step_end = timer()

            plot_start = timer()
            if any_plot:
                results.live_plot(
                    env=env,
                    time_step=time_step,
                    fig=fig,
                    ax=ax,
                    data=self.data,
                    sidebar=False,
                    separable=True,
                    map_distance=particle_distance,
                )
                # safe image buf
                tmp_buf = BytesIO()
                fig.savefig(tmp_buf, format="png", bbox_inches="tight")
                self.image_buf = tmp_buf
            plot_end = timer()

            particle_save_start = timer()
            for t in range(n_targets):
                np.save(
                    f'{results.logdir}/{self.data["utc_time"]}_target{t}_particles.npy',
                    env.pf[t].particles,
                )
            particle_save_end = timer()

            data_start = timer()
            with open(
                f"{results.logdir}/birdseye-{global_start_time}.log",
                "a",
                encoding="UTF-8",
            ) as outfile:
                json.dump(self.data, outfile, cls=NumpyEncoder)
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

            time_step += 1

        if self.config.get("make_gif", "false").lower() == "true":
            results.save_gif("tracking")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path", help="Path to config file, geolocate.ini provided as example."
    )
    parser.add_argument("--log", default="INFO", help="Log level")
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % args.log)
    logging.basicConfig(level=numeric_level, format="[%(asctime)s] %(message)s")
    logging.getLogger("matplotlib.font_manager").disabled = True

    instance = Geolocate(config_path=args.config_path)
    instance.start()
