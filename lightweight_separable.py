import argparse
from datetime import datetime
import configparser
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.spatial import distance
from tqdm.auto import trange

import birdseye.utils
import birdseye.sensor
import birdseye.actions
import birdseye.state
import birdseye.env
from birdseye.utils import tracking_metrics_separable


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
    for i in target_selections_not_found: 
    #for i in range(env.state.n_targets):
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

        #
        # print(f"{distances_to_other=}")
        # print(f"{np.argmin(distances_to_other)=}")
        # print(f"{np.argsort(distances_to_other)=}")
        # print(f"{target_proposals=}")
        # print(f"{sorted_proposals.shape=}")
        # print(f"{min_dist_proposal=}")
    # print(f"{proposals=}")
    # print(f"{not_found_sorted=}")

    #print(f"{env.get_particle_centroids()[object_of_interest]=}")
    if proposals:


        ####################

        # convert proposal end points to polar coordinates 
        proposals = {i:birdseye.utils.cart2pol(p[0],p[1]) for i,p in proposals.items()}
        #print(f"{proposals=}")

        # create trajectories of length=horizon from proposal end points (first turn, then move straight)
        trajectories = {}
        for i,p in proposals.items():
            trajectory = np.zeros((horizon,2))
            trajectory[:,1] = np.minimum(p[0]/horizon, env.state.sensor_speed)
            trajectory[0,0] = np.degrees(p[1])
            trajectories[i] = trajectory

        # check void probability contstraint for each trajectory, choose first that is sufficient 
        for i in std_dev_sorted:
            if i not in trajectories: 
                continue
            trj = trajectories[i]
        #for i,trj in trajectories.items():
            void_condition, particles = env.void_probability(trj, r_min, min_bound=min_bound)
            # if void_condition and (i in target_selections):
            #     target_selections.remove(i)
            #     if len(target_selections) == 0: 
            #         target_selections = {t for t in range(env.state.n_targets)}
            if void_condition: 
                target_selections.discard(i)
                if len(target_selections) == 0: 
                    target_selections = {t for t in range(env.state.n_targets)}
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


def main(config_path="lightweight_separable_config.ini"): 

    n_simulations = 100
    max_iterations = 400
    reward_func = lambda *args, **kwargs: None
    config_path = config_path
    
    r_min = 10
    horizon = 8
    min_bound = 0.82
    min_std_dev = 35
    num_particles = 3000

    config = configparser.ConfigParser()
    config.read(config_path)
    config = config["lightweight"]

    local_plot = config.get("native_plot", "false").lower()
    make_gif = config.get("make_gif", "false").lower()

    n_targets = int(config.get("n_targets", str(2)))
    antenna_type = config.get("antenna_type", "logp")
    planner_method = config.get("planner_method", "lightweight")
    experiment_name = config.get("experiment_name", planner_method)
    target_speed = float(config.get("target_speed", str(0.5)))
    sensor_speed = float(config.get("sensor_speed", str(1.0)))
    power_tx = [float(x) for x in config.get("power_tx", "26,26").split(",")]
    directivity_tx = [float(x) for x in config.get("directivity_tx", "1,1").split(",")]
    freq = [float(x) for x in config.get("freq", "5.7e9, 5.7e9").split(",")]
    fading_sigma = float(config.get("fading_sigma", str(8)))
    threshold = float(config.get("threshold", str(-120)))

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
            fig = plt.figure(figsize=(18, 10), dpi=50)
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
        actions = birdseye.actions.BaselineActions()
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
            sensor=sensor, actions=actions, state=state, simulated=True
        )

        belief = env.reset(num_particles=num_particles)

        #planner = birdseye.planner.LightweightPlanner(env, actions)

        target_selections = {t for t in range(n_targets)}
        control_actions = []
        #for i in range(max_iterations): 
        for i in trange(max_iterations, desc='Time steps'):
            if i%horizon == 0:
                if planner_method == "lightweight_simple":
                    control_action = get_control_actions(env, min_std_dev, r_min, horizon, min_bound)
                else:
                    control_action = get_control_actions_improved(env, min_std_dev, r_min, horizon, min_bound, target_selections)
                if control_action is None: 
                    # all objects localized 
                    break
                control_actions.extend(control_action)
            action = control_actions[i]
            #print(f"{action=}")
            (env_obs, reward, _, info) = env.step(action)

            if (local_plot == "true") or (make_gif == "true"):
                results.live_plot(
                    env=env, time_step=i, fig=fig, ax=ax, data={}
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
                "mae": mae
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
    #for i in range(n_simulations): 
        run_simulation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="lightweight_separable_config.ini")
    args = parser.parse_args()

    main(config_path=args.config_path)