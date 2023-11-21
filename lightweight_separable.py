import argparse
from datetime import datetime
import configparser
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.spatial import distance
from tqdm.auto import trange
from timeit import default_timer as timer
from multiprocessing import Process
import itertools

import birdseye.utils
import birdseye.sensor
import birdseye.actions
import birdseye.state
import birdseye.env
from birdseye.utils import tracking_metrics_separable, targets_found
from birdseye.planners.light_mcts import LightMCTS
from birdseye.planners.lavapilot import LAVAPilot
from birdseye.planners.repp import REPP


def main(config=None, config_path=None):
    n_simulations = 100
    max_iterations = 400
    reward_func = lambda pf: pf.weight_entropy  # lambda *args, **kwargs: None
    r_min = 10
    horizon = 1  # 8
    min_bound = 0.82
    min_std_dev = 35
    num_particles = 3000  # 3000

    default_config = {
        "native_plot": "false",
        "make_gif": "false",
        "n_targets": "2",
        "antenna_type": "logp",
        "planner_method": "lightweight",
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
    }
    if (config and config_path):
        raise ValueError("config and config_path cannot both be defined")

    if config_path:
        config = configparser.ConfigParser()
        config.read(config_path)
        config = config["lightweight"]
    elif config is None:
        config = default_config

    local_plot = config.get("native_plot", default_config["native_plot"]).lower()
    make_gif = config.get("make_gif", default_config["make_gif"]).lower()
    n_targets = int(config.get("n_targets", default_config["n_targets"]))
    antenna_type = config.get("antenna_type", default_config["antenna_type"])
    planner_method = config.get("planner_method", default_config["planner_method"])
    experiment_name = config.get("experiment_name", planner_method)
    target_speed = float(config.get("target_speed", default_config["target_speed"]))
    sensor_speed = float(config.get("sensor_speed", default_config["sensor_speed"]))
    
    power_tx = config.get("power_tx", default_config["power_tx"])
    power_tx = [float(x) for x in power_tx.split(",")]
    if len(power_tx) == 1:
        power_tx = [power_tx[0] for _ in range(n_targets)]
    
    directivity_tx = config.get("directivity_tx", default_config["directivity_tx"])
    directivity_tx = [float(x) for x in directivity_tx.split(",")]
    if len(directivity_tx) == 1:
        directivity_tx = [directivity_tx[0] for _ in range(n_targets)]
    
    freq = config.get("freq", default_config["freq"])
    freq = [float(x) for x in freq.split(",")]
    if len(freq) == 1:
        freq = [freq[0] for _ in range(n_targets)]
   
    fading_sigma = float(config.get("fading_sigma", default_config["fading_sigma"]))
    threshold = float(config.get("threshold", default_config["threshold"]))
    depth = int(config.get("depth", default_config["mcts_depth"]))
    c = float(config.get("c", default_config["mcts_c"]))
    mcts_simulations = int(
        config.get("mcts_simulations", default_config["mcts_simulations"])
    )
    n_downsample = int(config.get("n_downsample", default_config["mcts_n_downsample"]))

    # Sensor
    if antenna_type in ["directional", "yagi", "logp"]:
        antenna_filename = "radiation_pattern_yagi_5.csv"
    elif antenna_type in ["omni", "omnidirectional"]:
        antenna_filename = "radiation_pattern_monopole.csv"

    # BirdsEye
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # pylint: disable=no-member

    def run_simulation():
        global_start_time = datetime.utcnow().timestamp()

        results = birdseye.utils.Results(
            experiment_name=experiment_name,
            global_start_time=global_start_time,
            config=config,
        )

        if (local_plot == "true") or (make_gif == "true"):
            fig = plt.figure(figsize=(14, 10), dpi=100)
            ax = fig.subplots()
            fig.set_tight_layout(True)

        sensor = birdseye.sensor.SingleRSSISeparable(
            antenna_filename=antenna_filename,
            power_tx=power_tx,
            directivity_tx=directivity_tx,
            freq=freq,
            n_targets=n_targets,
            fading_sigma=fading_sigma,
        )

        actions = birdseye.actions.BaselineActions(sensor_speed=sensor_speed)
        # actions.print_action_info()

        state = birdseye.state.RFMultiState(
            n_targets=n_targets,
            target_speed=target_speed,
            sensor_speed=sensor_speed,
            reward=reward_func,
            simulated=True,
        )

        env = birdseye.env.RFMultiSeparableEnv(
            sensor=sensor,
            actions=actions,
            state=state,
            simulated=True,
            num_particles=num_particles,
        )
        env.reset()

        target_selections = {t for t in range(n_targets)}
        if planner_method == "repp":  # REPP
            planner = REPP(
                env, min_std_dev, r_min, horizon, min_bound, target_selections
            )
        elif planner_method == "lavapilot":  # LAVAPilot
            planner = LAVAPilot(env, min_std_dev, r_min, horizon, min_bound)
        elif planner_method == "mcts":  # MCTS
            planner = LightMCTS(
                env,
                depth=depth,
                c=c,
                simulations=mcts_simulations,
                n_downsample=n_downsample,
            )
        else:
            raise Exception

        control_actions = []

        for i in trange(max_iterations, desc="Time steps"):
            if i % horizon == 0:
                if targets_found(env, min_std_dev):
                    # all objects localized
                    control_action = None
                    break
                else:
                    plan_start_time = timer()
                    control_action = planner.get_action()
                    plan_end_time = timer()

                control_actions.extend(control_action)

            action = control_actions[i]
            # print(f"{action=}")
            (env_obs, reward, _, info) = env.step(action)

            if (local_plot == "true") or (make_gif == "true"):
                results.live_plot(
                    env=env, time_step=i, fig=fig, ax=ax, data={}, separable=True
                )

            (
                r_error,
                theta_error,
                heading_error,
                centroid_distance_error,
                rmse,
                mae,
            ) = tracking_metrics_separable(
                env.state.target_state, env.get_all_particles()
            )

            utc_time = datetime.utcnow().timestamp()
            # results.data_to_npy(env.get_all_particles(), "particles", utc_time)
            ### save results
            data = {
                "time": utc_time,
                "target": env.state.target_state,
                "sensor": env.state.sensor_state,
                "action": action,
                "observation": info["observation"],
                "std_dev_cartesian": env.get_particle_std_dev_cartesian(),
                "std_dev_polar": env.get_particle_std_dev_polar(),
                "r_err": r_error,
                "theta_err": theta_error,
                "heading_err": heading_error,
                "centroid_distance_err": centroid_distance_error,
                "rmse": rmse,
                "mae": mae,
                "plan_time": plan_end_time - plan_start_time,
            }
            results.data_to_json(data)

        if make_gif == "true":
            results.save_gif("tracking")

        if (local_plot == "true") or (make_gif == "true"):
            plt.close(fig)

    for i in trange(n_simulations, desc="Experiments"):
        run_simulation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_mode", action="store_true")
    parser.add_argument(
        "--config_path", type=str, default="lightweight_separable_config.ini"
    )
    args = parser.parse_args()

    if args.batch_mode:
        procs = []
        n_targets = [4, 8]
        target_speeds = [0.1, 0.5, 1]
        sensor_speeds = [1, 2, 3]
        planner_methods = ["repp", "lavapilot"]  # "mcts"
        fading_sigmas = [5, 10]
        for conf in list(
            itertools.product(
                n_targets, target_speeds, sensor_speeds, planner_methods, fading_sigmas
            )
        ):
            n_target, target_speed, sensor_speed, planner_method, fading_sigma = conf
            config = {
                "experiment_name": f"{planner_method}_{target_speed}targetspeed_{sensor_speed}sensorspeed_{n_target}targets_{fading_sigma}fading",
                "n_targets": str(n_target),
                "target_speed": str(target_speed),
                "sensor_speed": str(sensor_speed),
                "planner_method": planner_method,
                "fading_sigma": str(fading_sigma),
            }
            proc = Process(target=main, kwargs=({"config": config}))
            procs.append(proc)
            proc.start()

        # complete the processes
        for proc in procs:
            proc.join()

    else:
        main(config_path=args.config_path)
