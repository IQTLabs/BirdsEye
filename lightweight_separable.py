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
from birdseye.utils import tracking_metrics_separable
from light_mcts import LightMCTS


def circ_tangents(point, center, radius):
    px, py = point
    cx, cy = center

    b = np.sqrt((px-cx)**2 + (py-cy)**2)
    if radius >= b:
        ##print(f"Warning: No tangents are possible.  radius >= distance to circle center ({radius} >= {b})")
        return None
    th = np.arccos(radius / b)
    d = np.arctan2(py-cy, px-cx)
    d1 = d + th
    d2 = d - th

    tangents = [[cx+radius*np.cos(d1), cy+radius*np.sin(d1)],[cx+radius*np.cos(d2), cy+radius*np.sin(d2)]]
    v = np.array(point) - np.array(center) 
    v /= np.linalg.norm(v)
    intersection = np.array(center) + radius*v
    tangents.append(list(intersection))
    return np.array(tangents)

def get_control_actions(env, min_std_dev, r_min, horizon, min_bound):
    control_action = None
    std_dev = np.amax(env.get_particle_std_dev_cartesian(), axis=1) # get maximum standard deviation axis for each target
    #print(f"{std_dev=}")
    not_found = np.where(std_dev > min_std_dev)
    if len(not_found[0]):
        object_of_interest = np.argmin(std_dev[not_found])
    else:
        print(f"All objects localised!")
        return None
    #print(f"{object_of_interest=}")
    # get tangents and min distance point proposals
    proposals = circ_tangents([0,0], env.get_particle_centroids()[object_of_interest],  r_min)
    #print(f"{env.get_particle_centroids()[object_of_interest]=}")
    if proposals is not None:
        # convert proposal end points to polar coordinates 
        proposals = [birdseye.utils.cart2pol(p[0],p[1]) for p in proposals]
        #print(f"{proposals=}")

        # create trajectories of length=horizon from proposal end points (first turn, then move straight)
        trajectories = []
        for p in proposals:
            trajectory = np.zeros((horizon,2))
            trajectory[:,1] = np.minimum(p[0]/horizon, env.state.sensor_speed)
            trajectory[0,0] = np.degrees(p[1])
            trajectories.append(trajectory)

        # check void probability contstraint for each trajectory, choose first that is sufficient 
        for trj in trajectories:
            void_condition, particles = env.void_probability(trj, r_min, min_bound=min_bound)
            if void_condition:
                control_action = trj
                #print(f"Found good trajectory!")
                break
                
    deg_width = 40 
    default_controls = np.linspace(-180,int(180-deg_width),int(360/deg_width))

    # if no optimal trajectory meets void constraint 
    if control_action is None:

        # create trajectories from default controls 
        trajectories = []
        for c in default_controls:
            trajectory = np.zeros((horizon, 2))
            trajectory[:,1] = env.state.sensor_speed
            trajectory[0,0] = c
            trajectories.append(trajectory)

        # check void constraint for each trajectory 
        distances = []
        constrained_trajectories = []
        for trj in trajectories:
            void_condition, particles = env.void_probability(trj, r_min, min_bound=min_bound)
            if void_condition:
                constrained_trajectories.append(trj)
                distances.append(env.get_particle_centroids(particles=particles)[object_of_interest][0])
        # choose the sufficient trajectory that results in the min distance to centroid 
        if distances:
            control_action = constrained_trajectories[np.argmin(distances)]

    if control_action is None:
        print(f"Error: No path satisfies void constraints. Choosing random path.")
        control_action = trajectories[np.random.randint(len(default_controls))]
    return control_action

def get_control_actions_improved(env, min_std_dev, r_min, horizon, min_bound, target_selections):
    control_action = None
    std_dev = np.amax(env.get_particle_std_dev_cartesian(), axis=1) # get maximum standard deviation axis for each target
    #print(f"{std_dev=}")
    not_found = np.where(std_dev > min_std_dev)
    found = np.where(std_dev <= min_std_dev)
    target_selections_not_found = target_selections.copy()
    for f in found[0]: 
        target_selections_not_found.discard(f)
    if len(target_selections_not_found) == 0: 
        target_selections_not_found = {t for t in not_found[0]}
    std_dev_sorted = np.argsort(std_dev)
    not_found_sorted = np.argsort(std_dev[not_found])
    if len(not_found[0]):
        object_of_interest = not_found[0][np.argmin(std_dev[not_found])]
    else:
        print(f"All objects localised!")
        return None


    # get tangents and min distance point proposals
    #print(f"{env.get_particle_centroids()=}")
    centroids = env.get_particle_centroids()
    proposals = {}
    trajectories = {}
    #for i in target_selections_not_found: 
    #for i in range(env.state.n_targets):
    for i in target_selections:
        mean_other_centroids = [np.mean(np.delete(centroids,i,axis=0), axis=0)]
        target_proposals = circ_tangents([0,0], env.get_particle_centroids()[i],  r_min)
        # for p in proposals: 
        #     print(f"{p=}")
        #     print(f"{mean_other_centroids[0]=}")
        #     print(f"{np.linalg.norm(np.array(p)-mean_other_centroids[0])=}")
        # print(f"{i=}, {mean_other_centroids=}")
        # print(f"{i=}, {proposals=}")
        if target_proposals is not None: 
            distances_to_other = distance.cdist(mean_other_centroids,target_proposals)[0]
            sorted_proposals = target_proposals[np.argsort(distances_to_other)]
            min_dist_proposal = target_proposals[np.argmin(distances_to_other)]
            proposals[i] = min_dist_proposal

            proposals[i] = birdseye.utils.cart2pol(min_dist_proposal[0],min_dist_proposal[1])
            trajectory = np.zeros((horizon,2))
            trajectory[:,1] = np.minimum(proposals[i][0]/horizon, env.state.sensor_speed)
            trajectory[0,0] = np.degrees(proposals[i][1])
            trajectories[i] = trajectory

            void_condition, particles = env.void_probability(trajectory, r_min, min_bound=min_bound)
            # if void_condition and (i in target_selections):
            #     target_selections.remove(i)
            #     if len(target_selections) == 0: 
            #         target_selections = {t for t in range(env.state.n_targets)}
            if void_condition: 
                #target_selections.discard(i)
                target_selections.remove(i)
                if len(target_selections) == 0: 
                    target_selections.update([t for t in range(env.state.n_targets)])
                control_action = trajectory
                #print(f"Found good trajectory!")
                break

    # if proposals:
    #     ####################

    #     # convert proposal end points to polar coordinates 
    #     proposals = {i:birdseye.utils.cart2pol(p[0],p[1]) for i,p in proposals.items()}
    #     #print(f"{proposals=}")

    #     # create trajectories of length=horizon from proposal end points (first turn, then move straight)
    #     trajectories = {}
    #     for i,p in proposals.items():
    #         trajectory = np.zeros((horizon,2))
    #         trajectory[:,1] = np.minimum(p[0]/horizon, env.state.sensor_speed)
    #         trajectory[0,0] = np.degrees(p[1])
    #         trajectories[i] = trajectory

    #     # check void probability contstraint for each trajectory, choose first that is sufficient 
    #     # for i in std_dev_sorted:
    #     #     if i not in trajectories: 
    #     #         continue
    #     #     trj = trajectories[i]
    #     for i,trj in trajectories.items():
    #         void_condition, particles = env.void_probability(trj, r_min, min_bound=min_bound)
    #         # if void_condition and (i in target_selections):
    #         #     target_selections.remove(i)
    #         #     if len(target_selections) == 0: 
    #         #         target_selections = {t for t in range(env.state.n_targets)}
    #         if void_condition: 
    #             #target_selections.discard(i)
    #             target_selections.remove(i)
    #             if len(target_selections) == 0: 
    #                 target_selections = {t for t in range(env.state.n_targets)}
    #             control_action = trj
    #             #print(f"Found good trajectory!")
    #             break
                
    deg_width = 40 
    default_controls = np.linspace(-180,int(180-deg_width),int(360/deg_width))

    # if no optimal trajectory meets void constraint 
    if control_action is None:
        print('no optimal control')
        # create trajectories from default controls 
        trajectories = []
        for c in default_controls:
            trajectory = np.zeros((horizon, 2))
            trajectory[:,1] = env.state.sensor_speed
            trajectory[0,0] = c
            trajectories.append(trajectory)

        # check void constraint for each trajectory 
        distances = []
        constrained_trajectories = []
        for trj in trajectories:
            void_condition, particles = env.void_probability(trj, r_min, min_bound=min_bound)
            if void_condition:
                constrained_trajectories.append(trj)
                distances.append(env.get_particle_centroids(particles=particles)[object_of_interest][0])
        # choose the sufficient trajectory that results in the min distance to centroid 
        if distances:
            control_action = constrained_trajectories[np.argmin(distances)]

    if control_action is None:
        print(f"Error: No path satisfies void constraints. Choosing random path.")
        control_action = trajectories[np.random.randint(len(default_controls))]
    return control_action

def targets_found(env, min_std_dev):
    std_dev = np.amax(env.get_particle_std_dev_cartesian(), axis=1) # get maximum standard deviation axis for each target
    not_found = np.where(std_dev > min_std_dev)
    if len(not_found[0]) == 0:
        return True
    return False

def main(config=None, config_path=None): 

    n_simulations = 100
    max_iterations = 400
    reward_func = lambda pf: pf.weight_entropy #lambda *args, **kwargs: None    
    r_min = 10
    horizon = 1#8
    min_bound = 0.82
    min_std_dev = 35
    num_particles = 3000#3000

    default_config = ({
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
    })
    assert config is None or config_path is None, "config and config_path cannot both be defined"

    if config is None and config_path: 
        config = configparser.ConfigParser()
        config.read(config_path)
        config = config["lightweight"]

    config = default_config | config
    local_plot = config.get("native_plot", default_config["native_plot"]).lower()
    make_gif = config.get("make_gif", default_config["make_gif"]).lower()
    n_targets = int(config.get("n_targets", default_config["n_targets"]))
    antenna_type = config.get("antenna_type", default_config["antenna_type"])
    planner_method = config.get("planner_method", default_config["planner_method"])
    experiment_name = config.get("experiment_name", planner_method)
    target_speed = float(config.get("target_speed", default_config["target_speed"]))
    sensor_speed = float(config.get("sensor_speed", default_config["sensor_speed"]))
    if len(config.get("power_tx").split(",")) == 1: 
        config["power_tx"] = ",".join([config["power_tx"] for _ in range(n_targets)])
    power_tx = [float(x) for x in config.get("power_tx", ",".join(default_config["power_tx"] for _ in range(n_targets))).split(",")]
    if len(config.get("directivity_tx").split(",")) == 1: 
        config["directivity_tx"] = ",".join([config["directivity_tx"] for _ in range(n_targets)])
    directivity_tx = [float(x) for x in config.get("directivity_tx", ",".join(default_config["directivity_tx"] for _ in range(n_targets))).split(",")]
    if len(config.get("freq").split(",")) == 1: 
        config["freq"] = ",".join([config["freq"] for _ in range(n_targets)])
    freq = [float(x) for x in config.get("freq", ",".join(default_config["freq"] for _ in range(n_targets))).split(",")]
    fading_sigma = float(config.get("fading_sigma", default_config["fading_sigma"]))
    threshold = float(config.get("threshold", default_config["threshold"]))
    depth = int(config.get("depth", default_config["mcts_depth"]))
    c = float(config.get("c", default_config["mcts_c"]))
    mcts_simulations = int(config.get("mcts_simulations", default_config["mcts_simulations"]))
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

        # Action space
        actions = birdseye.actions.BaselineActions(sensor_speed=sensor_speed)
        #actions.print_action_info()

        # State managment
        state = birdseye.state.RFMultiState(
            n_targets=n_targets, 
            target_speed=target_speed, 
            sensor_speed=sensor_speed, 
            reward=reward_func, 
            simulated=True,
        )

        # Environment
        env = birdseye.env.RFMultiSeparableEnv(
            sensor=sensor, actions=actions, state=state, simulated=True, num_particles=num_particles
        )

        belief = env.reset()

        mcts = LightMCTS(env, depth=depth, c=c, simulations=mcts_simulations, n_downsample=n_downsample)

        #planner = birdseye.planner.LightweightPlanner(env, actions)

        target_selections = {t for t in range(n_targets)}
        control_actions = []
        #for i in range(max_iterations): 
        for i in trange(max_iterations, desc='Time steps'):
            if i%horizon == 0:
                plan_start_time = timer()
                if planner_method == "lavapilot": # LAVAPilot
                    control_action = get_control_actions(env, min_std_dev, r_min, horizon, min_bound)
                elif planner_method == "mcts": # mcts
                    if targets_found(env, min_std_dev): 
                        control_action = None
                    else: 
                        control_action = mcts.get_action()
                elif planner_method == "repp":
                    control_action = get_control_actions_improved(env, min_std_dev, r_min, horizon, min_bound, target_selections)
                else: 
                    raise Exception
                plan_end_time = timer()

                if control_action is None: 
                    # all objects localized 
                    break

                control_actions.extend(control_action)
            action = control_actions[i]
            #print(f"{action=}")
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
            ) = tracking_metrics_separable(env.state.target_state, env.get_all_particles())

            utc_time = datetime.utcnow().timestamp()
            #results.data_to_npy(env.get_all_particles(), "particles", utc_time)
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
                

        # print(f"{env.pf.particles.shape=}")
        # print(f"{env.get_particle_centroids(env.pf.particles)=}")
        # print(f"{env.get_target_std_dev()=}")
        # print(f"{np.argmin(env.get_target_std_dev())=},{env.get_particle_centroids(env.pf.particles)[np.argmin(env.get_target_std_dev())]=}")
        # #print(f"{circ_tangents(env.get_particle_centroids(env.pf.particles)[np.argmin(env.get_target_std_dev())], [0,0], 1)=}")
        # print(f"{circ_tangents([0,0], env.get_particle_centroids(env.pf.particles)[np.argmin(env.get_target_std_dev())],  1)=}")
        # print(f"{circ_tangents([0.,0.], [5,5],  5)=}")
        # print(f"{proposals=}")
        # print(f"{trajectories=}")
        # print(f"{env.void_probability(trajectories[0], r_min)=}")
        # print(f"{control_action=}")

    for i in trange(n_simulations, desc='Experiments'):
        run_simulation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_mode', action='store_true')
    parser.add_argument("--config_path", type=str, default="lightweight_separable_config.ini")
    args = parser.parse_args()

    if args.batch_mode:
        procs = []
        n_targets = [8]
        target_speeds = [0.1, 0.5, 1]
        sensor_speeds = [1, 2, 3]
        planner_methods = ["repp","lavapilot"] #"mcts"
        for conf in list(itertools.product(n_targets, target_speeds, sensor_speeds, planner_methods)): 
            n_target, target_speed, sensor_speed, planner_method = conf
            config = ({
                "experiment_name": f"{planner_method}_{target_speed}targetspeed_{sensor_speed}sensorspeed_{n_target}",
                "n_targets": str(n_target),
                "target_speed": str(target_speed),
                "sensor_speed": str(sensor_speed), 
                "planner_method": planner_method,
            }) 
            proc = Process(target=main, kwargs=({"config":config}))
            procs.append(proc)
            proc.start()

        # complete the processes
        for proc in procs:
            proc.join()

    else: 
        main(config_path=args.config_path)