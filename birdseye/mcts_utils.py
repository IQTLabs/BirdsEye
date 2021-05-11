# mcts_utils.py


# Imports
import random
import csv 
import numpy as np
from .utils import pol2cart, build_plots, tracking_error
from pfilter import ParticleFilter, systematic_resample

##################################################################
# MCTS Algorithm
##################################################################

def arg_max_action(actions, Q, N, history, c=None, exploration_bonus=False):

    # only need to compute if exploration possibility
    if exploration_bonus:
        N_h = 0
        for action in actions.get_action_list():
            new_index = history.copy()
            new_index.append(action)
            N_h += N[tuple(new_index)]

    values = []
    for action in actions.get_action_list():
        new_index = history.copy()
        new_index.append(action)

        Q_val = Q[tuple(new_index)]

        # best action with exploration possibility
        if exploration_bonus:
            if N[tuple(new_index)] == 0:
                return action

            # compute exploration bonus, checking for zeroes (I don't think this will ever occur anyway...)
            log_N_h = np.log(N_h)
            if log_N_h < 0:
                log_N_h = 0

            numerator = np.sqrt(log_N_h)
            denominator = N[tuple(new_index)]
            exp_bonus = c * numerator / denominator

            Q_val += exp_bonus

        values.append(Q_val)

    return np.argmax(values)


##################################################################
# Rollout
##################################################################
def rollout_random(env, state, depth):

    if depth == 0:
        return 0

    ## random action
    action, action_index = env.actions.get_random_action()

    # generate next state and reward with random action; observation doesn't matter
    state_prime = env.state.update_state(state, action)
    reward = env.state.reward_func(state=state_prime, action_idx=action_index, particles=env.pf.particles)

    return reward + lambda_arg * rollout_random(env, state_prime, depth-1)


##################################################################
# Simulate
##################################################################
def simulate(env, Q, N, state, history, depth, c):

    if depth == 0:
        return (Q, N, 0)

    # expansion
    test_index = history.copy()
    test_index.append(1)

    if tuple(test_index) not in Q:

        for action in env.actions.get_action_list():
            # initialize Q and N to zeros
            new_index = history.copy()
            new_index.append(action)
            Q[tuple(new_index)] = 0
            N[tuple(new_index)] = 0

        # rollout
        return (Q, N, rollout_random(env, state, depth))

    # search: find optimal action to explore
    search_action_index = arg_max_action(env.actions, Q, N, history, c, True)
    action = env.actions.index_to_action(search_action_index)

    # take action; get new state, observation, and reward
    state_prime = env.state.update_state(state, action)
    observation = env.sensor.observation(state_prime)
    reward = env.state.reward_func(state=state_prime, action_idx=search_action_index, particles=env.pf.particles)

    # recursive call after taking action and getting observation
    new_history = history.copy()
    new_history.append(search_action_index)
    new_history.append(observation)
    (Q, N, successor_reward) = simulate(env, Q, N, state_prime, new_history, depth-1, c)
    q = reward + lambda_arg * successor_reward

    # update counts and values
    update_index = history.copy()
    update_index.append(search_action_index)
    N[tuple(update_index)] += 1
    Q[tuple(update_index)] += ((q - Q[tuple(update_index)]) / N[tuple(update_index)])

    return (Q, N, q)


##################################################################
# Select Action
##################################################################
def select_action(env, Q, N, belief, depth, c, iterations):

    # empty history at top recursive call
    history = []

    # number of iterations
    counter = 0
    while counter < iterations:

        # draw state randomly based on belief state (pick a random particle)
        state = random.choice(belief)

        # simulate
        simulate(env, Q, N, state.astype(float), history, depth, c)

        counter += 1

    best_action_index = arg_max_action(env.actions, Q, N, history)
    action = env.actions.index_to_action(best_action_index)
    return (Q, N, action)


##################################################################
# Trial
##################################################################
lambda_arg = 0.95
def mcts_trial(env, num_iters, depth, c, plotting=False, simulations=1000, fig=None, ax=None):

    # Initialize true state and belief state (particle filter);
    # we assume perfect knowledge at start of simulation (could experiment otherwise with random beliefs)
    # state is [range, bearing, relative course, own speed]
    # assume a starting position within range of sensor and not too close
    env.reset()

    belief = env.pf.particles

    # global Q and N dictionaries, indexed by history (and optionally action to follow all in same array; using ints)
    ##Q = Dict{Array{Int64,1},Float64}()
    Q = {}
    ##N = Dict{Array{Int64,1},Float64}()
    N = {}

    # experimenting with different parameter values
    # experiment with different depth parameters
    depth = depth
    # exploration factor, experiment with different values
    c = c

    # don't need to modify history tree at first time step
    action = None
    observation = None

    # run simulation

    total_reward = 0
    total_col = 0
    total_loss = 0
    r_err_log = []
    theta_err_log = []
    heading_err_log = []
    centroid_err_log = []
    rmse_log = []

    # 500 time steps with an action to be selected at each
    plots = []

    log_file = open('employee_file2.csv', mode='w')
    fieldnames = ['time', 'dept', 'birth_month']
    log_writer = csv.DictWriter(log_file, fieldnames=fieldnames)

    for time_step in range(num_iters):

        #if time_step % 100 == 0
        #    @show time_step
        #end

        # NOTE: we found restarting history tree at each time step yielded better results
        # if action taken, modify history tree
        if action is not None:
            Q = {}
            N = {}

        # select an action
        (Q, N, action) = select_action(env, Q, N, belief, depth, c, simulations)

        # take action; get next true state, obs, and reward
        next_state = env.state.update_state(env.state.target_state, action, target_update=True)
        #next_state = env.state.update_state(env.state.target_state, action, target_control=env.state.circular_control(time_step, size=5))
        # Update absolute position of sensor 
        env.state.update_sensor(action)
        observation = env.sensor.observation(next_state)
        #print('true_state = {}, next_state = {}, action = {}, observation = {}'.format(env.state.target_state, next_state, action, observation))
        reward = env.state.reward_func(state=next_state, action_idx=env.actions.action_to_index(action), particles=env.pf.particles)
        env.state.target_state = next_state

        # pfrnn
        #env.pfrnn.update(observation, env.get_absolute_target(), env.actions.action_to_index(action))

        # update belief state (particle filter)
        env.pf.update(np.array(observation), xp=belief, control=action)
        belief = env.pf.particles

        # error metrics 
        r_error, theta_error, heading_error, centroid_distance_error, rmse  = tracking_error(env.state.target_state, env.pf.particles)
        r_err_log.append(r_error)
        theta_err_log.append(theta_error)
        heading_err_log.append(heading_error)
        centroid_err_log.append(centroid_distance_error)
        rmse_log.append(rmse)
        
        #r_error, theta_error, heading_error, centroid_distance_error, rmse  = tracking_error(env.get_absolute_target(), env.get_absolute_particles())

        # accumulate reward
        total_reward += reward
        
        if env.state.target_state[0] < 10:
            total_col = 1

        if env.state.target_state[0] > 150:
            total_loss = 1

        if plotting:
            build_plots(env.state.target_state, belief, env.state.sensor_state, env.get_absolute_target(), env.get_absolute_particles(), time_step, fig, ax)

        # TODO: flags for collision, lost track, end of simulation lost track

    return [plots, total_reward, total_col, total_loss, r_err_log, theta_err_log, heading_err_log, centroid_err_log, rmse_log]
    