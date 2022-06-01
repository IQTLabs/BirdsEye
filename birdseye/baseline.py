from datetime import datetime
import random
import configparser
import argparse
from types import SimpleNamespace
from tqdm import tqdm
from .actions import *
from .sensor import *
from .state import *
from .definitions import *
from .env import RFEnv
from .utils import write_header_log, Results, pol2cart, tracking_error, particle_swap

# Default baseline inputs
baseline_defaults = {
    'plotting' : False,
    'trials' : 500,
    'timesteps' : 150
}


def static(env):
    return (0,0)

def random_policy(env):
    random_action_index = random.choice(env.actions.get_action_list())
    return env.actions.index_to_action(random_action_index)

baseline_policy = {
    'static': static,
    'random': random_policy
}

def baseline_trial(env, policy, num_timesteps, plotting=False, results=None):

    # Initialize true state and belief state (particle filter);
    # we assume perfect knowledge at start of simulation (could experiment otherwise with random beliefs)
    # state is [range, bearing, relative course, own speed]
    # assume a starting position within range of sensor and not too close
    env.reset()

    belief = env.pf.particles


    # don't need to modify history tree at first time step
    action = None
    observation = None


    total_col = 0
    total_loss = 0

    # Save values for all iterations and episodes
    all_target_states = [None]*num_timesteps
    all_sensor_states = [None]*num_timesteps
    all_actions = [None]*num_timesteps
    all_obs = [None]*num_timesteps
    all_reward = np.zeros(num_timesteps)
    all_col = np.zeros(num_timesteps)
    all_loss = np.zeros(num_timesteps)
    all_r_err = np.zeros((num_timesteps, env.state.n_targets))
    all_theta_err = np.zeros((num_timesteps, env.state.n_targets))
    all_heading_err = np.zeros((num_timesteps, env.state.n_targets))
    all_centroid_err = np.zeros((num_timesteps, env.state.n_targets))
    all_rmse = np.zeros((num_timesteps, env.state.n_targets))
    all_mae = np.zeros((num_timesteps, env.state.n_targets))
    all_inference_times = np.zeros(num_timesteps)
    all_pf_cov = [None]*num_timesteps


    abs_particle_hist = []
    abs_target_hist = []

    # 500 time steps with an action to be selected at each
    plots = []

    for time_step in tqdm(range(num_timesteps)):

        # select an action
        inference_start_time = datetime.now()

        # action
        action = policy(env)

        inference_time = (datetime.now() - inference_start_time).total_seconds()
        # take action; get next true state, obs, and reward
        next_state = np.array([env.state.update_state(target_state, action) for target_state in env.state.target_state])
        # Update absolute position of sensor
        env.state.update_sensor(action)
        observation = env.sensor.observation(next_state)

        # pfrnn
        #env.pfrnn.update(observation, env.get_absolute_target(), env.actions.action_to_index(action))

        # update belief state (particle filter)
        env.pf.update(np.array(observation), xp=belief, control=action)
        particle_swap(env)
        belief = env.pf.particles
        #reward = env.state.reward_func(state=next_state, action_idx=env.actions.action_to_index(action), particles=env.pf.particles)
        reward = 0
        env.state.target_state = next_state

        # error metrics
        r_error, theta_error, heading_error, centroid_distance_error, rmse, mae  = tracking_error(env.state.target_state, env.pf.particles)

        total_col = np.mean([np.mean(env.pf.particles[:,4*t] < 15) for t in range(env.state.n_targets)])
        total_loss = np.mean([np.mean(env.pf.particles[:,4*t] > 150) for t in range(env.state.n_targets)])
        # for target_state in env.state.target_state:
        #     if target_state[0] < 10:
        #         total_col += 1

        #     if target_state[0] > 150:
        #         total_loss += 1

        if results is not None and results.plotting:
            results.build_multitarget_plots(env, time_step=time_step, centroid_distance_error=centroid_distance_error, selected_plots=[4])

        # Save results to output arrays
        all_target_states[time_step] = env.state.target_state
        all_sensor_states[time_step] = env.state.sensor_state
        all_actions[time_step] = action
        all_obs[time_step] = observation
        all_r_err[time_step] = r_error
        all_theta_err[time_step] = theta_error
        all_heading_err[time_step] = heading_error
        all_centroid_err[time_step] = centroid_distance_error
        all_rmse[time_step] = rmse
        all_mae[time_step] = mae
        all_reward[time_step] = reward
        all_col[time_step] = total_col
        all_loss[time_step] = total_loss
        all_inference_times[time_step] = inference_time
        all_pf_cov[time_step] = list(env.pf.cov_state.flatten())

        # TODO: flags for collision, lost track, end of simulation lost track

    return [plots, all_target_states, all_sensor_states, all_actions,
            all_obs, all_reward, all_col, all_loss, all_r_err,
            all_theta_err, all_heading_err, all_centroid_err, all_rmse, all_mae, all_inference_times, all_pf_cov]

def run_baseline(env, config=None, global_start_time=None):
    """Function to run Monte Carlo Tree Search

    Parameters
    ----------
    env : object
        Environment definitions
    config : object
        Config object which must have following:

    simulations : int
        Number of simulations
    DEPTH : int
        Tree depth
    lambda_arg : float
        Lambda value
    num_trials : int
        Number of trials
    timesteps : int
        Number of timesteps
    COLLISION_REWARD : float
        Reward value for collision
    LOSS_REWARD : float
        Reward value for loss function
    plotting : bool
        Flag to plot or not
    ------------
    fig : object
        Figure object
    ax : object
        Axis object
    """
    if config is None:
        config = SimpleNamespace(**baseline_defaults)
    # simulations = config.simulations
    # DEPTH = config.depth
    # lambda_arg = config.lambda_arg
    num_trials = config.trials
    timesteps = config.timesteps
    policy = baseline_policy[config.policy]
    # COLLISION_REWARD = config.collision
    # LOSS_REWARD = config.loss
    plotting = config.plotting

    # Results instance for saving results to file
    results = Results(method_name='baseline',
                        global_start_time=global_start_time,
                        num_iters=num_trials,
                        plotting=plotting)


    run_data = []

    lost = 0
    coll = 0
    for i in range(1, num_trials+1):
        run_start_time = datetime.now()
        result = baseline_trial(env, policy, timesteps, plotting, results=results)
        run_time = datetime.now()-run_start_time
        run_data.append([datetime.now(), run_time] + result[1:])

        coll = result[6][-1]
        lost = result[7][-1]
        print(".")
        print("\n==============================")
        print("Trial: {}".format(i))
        print("Collision Rate: {}".format(coll))
        print("Loss Rate: {}".format(lost))
        print("==============================")

        # Saving results to CSV file
        results.write_dataframe(run_data=run_data)
        if results.plotting:
            results.save_gif(i)



def baseline(args=None, env=None):
    defaults = baseline_defaults
    config = None

    if args:
        config = configparser.ConfigParser(defaults)  # pytype: disable=wrong-arg-types
        config.read_dict({section: dict(args[section]) for section in args.sections()})
        defaults = dict(config.items('Defaults'))
        # Fix for boolean args
        defaults['plotting'] = config.getboolean('Defaults', 'plotting')

    parser = argparse.ArgumentParser(description='Baselines',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_defaults(**defaults)
    # parser.add_argument('--lambda_arg', type=float, help='Lambda value')
    # parser.add_argument('--collision', type=float, help='Reward value for collision')
    # parser.add_argument('--loss', type=float, help='Reward value for loss function')
    # parser.add_argument('--depth', type=float, help='Tree depth')
    # parser.add_argument('--simulations', type=int, help='Number of simulations')
    parser.add_argument('--policy', type=str, help='Policy for actions')
    parser.add_argument('--plotting', type=bool, help='Flag to plot or not')
    parser.add_argument('--trials', type=int, help='Number of runs')
    parser.add_argument('--timesteps', type=int, help='Number of timesteps')
    args,_ = parser.parse_known_args()

    if not env:
        # Setup environment
        actions = SimpleActions()
        sensor = Drone()
        state = RFState()
        env = RFEnv(sensor, actions, state)

    global_start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if config:
        write_header_log(config, 'baseline', global_start_time)

    env.actions = BaselineActions()
    run_baseline(env=env, config=args, global_start_time=global_start_time)


if __name__ == '__main__':
    baseline()
