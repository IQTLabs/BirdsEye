# mcts_utils.py
# Imports
import random
from datetime import datetime
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .utils import particle_swap
from .utils import tracking_error

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
            log_N_h = max(log_N_h, 0)

            numerator = np.sqrt(log_N_h)
            denominator = N[tuple(new_index)]
            exp_bonus = c * numerator / denominator

            Q_val += exp_bonus

        values.append(Q_val)

    return np.argmax(values)


##################################################################
# Rollout
##################################################################
def rollout_random(env, state, depth, pf_copy):
    # print(f"Rollout: {depth=}")
    if depth == 0:
        return 0

    # random action
    action, action_index = env.actions.get_random_action()

    # print([state[4*t:4*(t+1)] for t in range(env.state.n_targets)])
    # generate next state and reward with random action; observation doesn't matter
    # state_prime = np.array([env.state.update_state(state[4*t:4*(t+1)], action) for t in range(env.state.n_targets)])
    next_state = np.array([env.state.update_sim_state(s, action) for s in state])
    rewards = []
    observations = []
    for t in range(env.state.n_targets):
        # Get sensor observation
        observation = env.sensor.observation(next_state[t], t)
        observations.append(observation)
        # Update particle filter
        pf_copy[t].update(
            np.array(observation), xp=pf_copy[t].particles, control=action
        )
        rewards.append(
            env.state.reward_func(
                pf=pf_copy[t],
                state=next_state,
                action_idx=action_index,
                particles=pf_copy[t].particles,
            )
        )
        # print(f"{pf_copy[t].n_eff=}, {pf_copy[t].n_eff_threshold=}")
        # print(f"{pf_copy[t].weight_entropy}")
    # reward = env.state.reward_func(
    #     state=next_state, action_idx=action_index, particles=pf_copy
    # )
    reward = np.mean(rewards)

    return reward + lambda_arg * rollout_random(env, next_state, depth - 1, pf_copy)


##################################################################
# Simulate
##################################################################
def simulate(env, Q, N, state, history, depth, c, pf_copy):
    # print(f"Simulate: {history=}, {depth=} \n {Q=}")

    if depth == 0:
        return (Q, N, 0)

    # expansion
    new_tree = history.copy()
    new_tree.append(
        np.random.randint(len(env.actions.get_action_list()))
    )  # TODO: why 1?

    if tuple(new_tree) not in Q:
        expansion_start = timer()
        # Q[tuple(new_tree)] = 0
        # N[tuple(new_tree)] = 0
        for action in env.actions.get_action_list():
            # initialize Q and N to zeros
            new_index = history.copy()
            new_index.append(action)
            Q[tuple(new_index)] = 0
            N[tuple(new_index)] = 0
        ret = rollout_random(env, state, depth, pf_copy)
        expansion_end = timer()

        # print(f"expansion time = {expansion_end-expansion_start}")

        return (Q, N, ret)
        # rollout
        # return (Q, N, rollout_random(env, state, depth, pf_copy))

    select_start = timer()
    # search: find optimal action to explore
    search_action_index = arg_max_action(env.actions, Q, N, history, c, True)

    action = env.actions.index_to_action(search_action_index)
    select_end = timer()

    # print(f"selection time = {select_end-select_start}")

    # print("Selected action = ",search_action_index)

    # take action; get new state, observation, and reward
    # state_prime = np.array([env.state.update_state(state[4*t:4*(t+1)], action) for t in range(env.state.n_targets)]) # env.state.update_state(state, action)

    state_start = timer()
    next_state = np.array([env.state.update_sim_state(s, action) for s in state])
    state_end = timer()

    # print(f"state time = {state_end-state_start}")

    # pf_start = timer()
    # o_start = timer()
    # observations = [env.sensor.observation(next_state[t], t)[0] for t in range(env.state.n_targets)]
    # o_end = timer()
    # p_start = timer()
    # for t in range(env.state.n_targets):
    #     pf_copy[t].update(np.array(observations[t]), control=action) #for t in range(env.state.n_targets)
    # p_end = timer()
    # r_start = timer()
    # reward = np.mean([env.state.reward_func(pf_copy[t]) for t in range(env.state.n_targets)])
    # r_end = timer()
    # pf_end = timer()
    # print(f"r time = {r_end-r_start}")
    # print(f"o time = {o_end-o_start}")
    # print(f"p time = {p_end-p_start}")
    # print(f"pf time = {pf_end-pf_start}")
    observations = []
    rewards = 0
    pf_start = timer()
    for t in range(env.state.n_targets):
        # Get sensor observation
        o_start = timer()
        observation = env.sensor.observation(next_state[t], t)[0]
        observations.append(observation)
        o_end = timer()
        # Update particle filter
        p_start = timer()
        pf_copy[t].update(
            np.array(observation), xp=pf_copy[t].particles, control=action
        )
        p_end = timer()
        r_start = timer()
        # rewards.append(env.state.reward_func(pf_copy[t]))
        rewards += env.state.reward_func(
            pf=pf_copy[t],
            state=next_state,
            action_idx=search_action_index,
            particles=pf_copy[t].particles,
        )

        r_end = timer()
    pf_end = timer()
    reward = rewards / env.state.n_targets

    # print(f"r time = {r_end-r_start}")
    # print(f"o time = {o_end-o_start}")
    # print(f"p time = {p_end-p_start}")
    # print(f"pf time = {pf_end-pf_start}")

    # if env.state.belief_mdp:
    #     env.pf.particles = belief
    #     env.pf.update(np.array(observation), xp=belief, control=action)
    #     belief = env.pf.particles

    # reward = env.state.reward_func(
    #     state=state_prime, action_idx=search_action_index, particles=belief
    # )
    # recursive call after taking action and getting observation
    tree_start = timer()
    new_history = history.copy()
    new_history.append(search_action_index)
    # new_history.append(tuple([int(o) for o in observations]))
    (Q, N, successor_reward) = simulate(
        env, Q, N, next_state, new_history, depth - 1, c, pf_copy
    )
    q = reward + lambda_arg * successor_reward

    # update counts and values
    update_index = history.copy()
    update_index.append(search_action_index)
    N[tuple(update_index)] += 1
    Q[tuple(update_index)] += (q - Q[tuple(update_index)]) / N[tuple(update_index)]
    tree_end = timer()

    # print(f"tree time = {tree_end-tree_start}")

    # print("Q update = ",Q[tuple(update_index)])
    return (Q, N, q)


##################################################################
# Select Action
##################################################################
def select_action(env, Q, N, belief, depth, c, iterations):
    # empty history at top recursive call
    history = []

    # number of iterations
    counter = 0

    original_particles = np.copy(env.pf.particles)
    original_n_particles = env.pf.n_particles
    original_weights = env.pf.weights
    n_particle_downsample = 100
    env.pf.n_particles = n_particle_downsample
    env.pf.weights = np.ones(env.pf.n_particles) / env.pf.n_particles
    while counter < iterations:
        # draw state randomly based on belief state (pick a random particle)
        state = random.choice(belief)
        converted_state = state.reshape(env.state.n_targets, 4)
        # simulate
        simulate(
            env,
            Q,
            N,
            converted_state.astype(float),
            history,
            depth,
            c,
            np.copy(original_particles)[
                random.sample(range(len(original_particles)), n_particle_downsample)
            ],
        )

        counter += 1
    env.pf.n_particles = original_n_particles
    env.pf.particles = original_particles
    env.pf.weights = original_weights
    best_action_index = arg_max_action(env.actions, Q, N, history)
    action = env.actions.index_to_action(best_action_index)
    return (Q, N, action)


def select_action_light(
    env, Q={}, N={}, depth=2, c=20, iterations=100, n_downsample=500
):
    # empty history at top recursive call
    history = []

    # number of iterations
    counter = 0

    while counter < iterations:
        # print(f"{counter}/{iterations} simulations")
        pf_copy = env.pf_copy(n_downsample=n_downsample)
        # draw state randomly based on belief state (pick a random particle)
        state = env.random_state(pf_copy)
        # converted_state = state.reshape(env.state.n_targets, 4)
        # simulate
        simulate(
            env,
            Q,
            N,
            state,
            history,
            depth,
            c,
            pf_copy,
        )

        counter += 1
    # print(f"{Q=}")
    # print(f"{N=}")
    best_action_index = arg_max_action(env.actions, Q, N, history)
    action = env.actions.index_to_action(best_action_index)
    return (Q, N, action)


def trim_tree(Q, N, action):
    # debug: assert Q.keys() == N.keys()
    for k in list(Q):
        if len(k) <= 1 or k[0] != action:
            Q.pop(k)
            N.pop(k)
        else:
            Q[(k[1],)] = Q.pop(k)
            N[(k[1],)] = N.pop(k)


class MCTSRunner:
    def __init__(self, env, depth, c, simulations=1000):
        self.env = env
        self.depth = depth
        self.c = c
        self.simulations = simulations

        self.Q = {}
        self.N = {}

        self.action = None

    def run(self, belief_heatmap):
        self.env.reset()

        if self.action is not None:
            self.Q = {}
            self.N = {}

        # self.Q, self.N, self.action = select_action(
        #     self.env,
        #     self.Q,
        #     self.N,
        #     self.env.pf.particles,
        #     self.depth,
        #     self.c,
        #     self.simulations,
        # )
        self.Q, self.N, self.action = select_action_light(
            self.env,
            self.Q,
            self.N,
            self.depth,
            self.c,
            self.simulations,
        )

        return self.action


##################################################################
# Trial
##################################################################
lambda_arg = 0.95


def mcts_trial(
    env,
    num_iters,
    depth,
    c,
    plotting=False,
    simulations=1000,
    fig=None,
    ax=None,
    results=None,
):
    # Initialize true state and belief state (particle filter);
    # we assume perfect knowledge at start of simulation (could experiment otherwise with random beliefs)
    # state is [range, heading, relative course, own speed]
    # assume a starting position within range of sensor and not too close
    env.reset()

    belief = env.pf.particles

    # global Q and N dictionaries, indexed by history (and optionally action to follow all in same array; using ints)
    # Q = Dict{Array{Int64,1},Float64}()
    Q = {}
    # N = Dict{Array{Int64,1},Float64}()
    N = {}

    # experimenting with different parameter values
    # experiment with different depth parameters
    depth = depth
    # exploration factor, experiment with different values
    c = c

    # don't need to modify history tree at first time step
    action = None
    observation = None

    total_col = 0
    total_loss = 0

    # Save values for all iterations and episodes
    all_target_states = [None] * num_iters
    all_sensor_states = [None] * num_iters
    all_actions = [None] * num_iters
    all_obs = [None] * num_iters
    all_reward = np.zeros(num_iters)
    all_col = np.zeros(num_iters)
    all_loss = np.zeros(num_iters)
    all_r_err = np.zeros((num_iters, env.state.n_targets))
    all_theta_err = np.zeros((num_iters, env.state.n_targets))
    all_heading_err = np.zeros((num_iters, env.state.n_targets))
    all_centroid_err = np.zeros((num_iters, env.state.n_targets))
    all_rmse = np.zeros((num_iters, env.state.n_targets))
    all_mae = np.zeros((num_iters, env.state.n_targets))
    all_inference_times = np.zeros(num_iters)
    all_pf_cov = [None] * num_iters

    # 500 time steps with an action to be selected at each
    plots = []
    selected_plots = [7]
    fig = plt.figure(figsize=(10 * len(selected_plots), 10), dpi=100)
    axs = None

    for time_step in tqdm(range(num_iters)):
        # if time_step % 100 == 0
        #    @show time_step
        # end

        # NOTE: we found restarting history tree at each time step yielded better results
        # if action taken, modify history tree
        if action is not None:
            Q = {}
            N = {}

        # select an action
        inference_start_time = datetime.now()

        (Q, N, action) = select_action(env, Q, N, belief, depth, c, simulations)
        inference_time = (datetime.now() - inference_start_time).total_seconds()
        # take action; get next true state, obs, and reward
        # next_state = env.state.update_state(env.state.target_state, action, target_update=True)
        next_state = np.array(
            [
                env.state.update_state(target_state, action)
                for target_state in env.state.target_state
            ]
        )
        # next_state = env.state.update_state(env.state.target_state, action, target_control=env.state.circular_control(time_step, size=5))
        # Update absolute position of sensor
        env.state.update_sensor(action)
        observation = env.sensor.observation(next_state)
        # print('true_state = {}, next_state = {}, action = {}, observation = {}'.format(env.state.target_state, next_state, action, observation))

        # pfrnn
        # env.pfrnn.update(observation, env.get_absolute_target(), env.actions.action_to_index(action))

        # update belief state (particle filter)
        env.pf.update(np.array(observation), xp=belief, control=action)
        # print(env.pf.particles.shape)
        particle_swap(env)
        belief = env.pf.particles

        reward = env.state.reward_func(
            state=next_state,
            action_idx=env.actions.action_to_index(action),
            particles=env.pf.particles,
        )
        env.state.target_state = next_state

        # error metrics
        (
            r_error,
            theta_error,
            heading_error,
            centroid_distance_error,
            rmse,
            mae,
        ) = tracking_error(env.state.target_state, env.pf.particles)

        # r_error, theta_error, heading_error, centroid_distance_error, rmse  = tracking_error(env.get_absolute_target(), env.get_absolute_particles())
        total_col = np.mean(
            [
                np.mean(env.pf.particles[:, 4 * t] < 15)
                for t in range(env.state.n_targets)
            ]
        )
        total_loss = np.mean(
            [
                np.mean(env.pf.particles[:, 4 * t] > 150)
                for t in range(env.state.n_targets)
            ]
        )

        # for target_state in env.state.target_state:
        #     if target_state[0] < 15:
        #         total_col += 1

        #     if target_state[0] > 150:
        #         total_loss += 1

        if results is not None and results.plotting:
            axs = results.build_multitarget_plots(
                env,
                time_step,
                fig,
                axs,
                centroid_distance_error,
                selected_plots=selected_plots,
            )

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

    return [
        plots,
        all_target_states,
        all_sensor_states,
        all_actions,
        all_obs,
        all_reward,
        all_col,
        all_loss,
        all_r_err,
        all_theta_err,
        all_heading_err,
        all_centroid_err,
        all_rmse,
        all_mae,
        all_inference_times,
        all_pf_cov,
    ]
