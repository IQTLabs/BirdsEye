import birdseye.utils
import birdseye.sensor
import birdseye.actions
import birdseye.state
import birdseye.env
import birdseye.mcts_utils
from sys import getsizeof
import numpy as np 
from timeit import default_timer as timer


class LightMCTS:
    def __init__(self, env, depth=3, c=20, simulations=100, n_downsample=400):

        self.env = env
        self.depth = depth
        self.c = c
        self.simulations = simulations
        self.n_downsample = n_downsample

        self.Q = {}
        self.N = {}

        self.action = None

    def get_action(self,):

        self.Q, self.N, self.action = birdseye.mcts_utils.select_action_light(
            self.env,
            self.Q,
            self.N,
            self.depth,
            self.c,
            self.simulations,
            self.n_downsample,
        )
        birdseye.mcts_utils.trim_tree(self.Q, self.N, self.env.actions.action_to_index(self.action))

        return [list(self.action)]

# n_targets = 2

# antenna_filename = "radiation_pattern_yagi_5.csv"
# power_tx = [26,26]
# directivity_tx = [1,1]
# freq = [5.7e9, 5.7e9]
# fading_sigma = 8
# target_speed = 0.5
# sensor_speed = 1.0
# reward_func = lambda pf: pf.weight_entropy

# sensor = birdseye.sensor.SingleRSSISeparable(
#     antenna_filename=antenna_filename,
#     power_tx=power_tx,
#     directivity_tx=directivity_tx,
#     freq=freq,
#     n_targets=n_targets,
#     fading_sigma=fading_sigma,
# )

# # Action space
# actions = birdseye.actions.BaselineActions()
# #actions.print_action_info()

# # State managment
# state = birdseye.state.RFMultiState(
#     n_targets=n_targets, 
#     target_speed=target_speed, 
#     sensor_speed=sensor_speed, 
#     reward=reward_func, 
#     simulated=True,
# )
# # Environment
# env = birdseye.env.RFMultiSeparableEnv(
#     sensor=sensor, actions=actions, state=state, simulated=True, num_particles=250
# )
# env.reset()

# print(actions.get_action_list())
# print(actions.avail_actions())

# new_pf = env.pf_copy()

# start = timer() 
# Q={}
# N={}
# Q,N,a = birdseye.mcts_utils.select_action_light(env, Q=Q, N=N)
# print(f"{Q=}")
# print(f"{N=}")
# print(f"{a=}")
# start_act = timer() 
# action = env.actions.action_to_index(a)
# end_act = timer()
# print(f"{action=}")
# print(f"action convert: {end_act-start_act}")
# start_trim = timer() 
# birdseye.mcts_utils.trim_tree(Q, N, action)
# end_trim = timer() 
# print(f"{Q=}")
# print(f"{N=}")
# print(f"trim: {end_trim-start_trim}")
# end = timer() 
# print(f"time = {end-start}")
# start = timer()
# (env_obs, reward, _, info) = env.step(a)
# end = timer() 
# print(f"step: {end-start}")


# start = timer() 
# Q,N,a = birdseye.mcts_utils.select_action_light(env, Q=Q, N=N)
# print(f"{Q=}")
# print(f"{N=}")
# print(f"{a=}")
# start_act = timer() 
# action = env.actions.action_to_index(a)
# end_act = timer()
# print(f"{action=}")
# print(f"action convert: {end_act-start_act}")
# start_trim = timer() 
# birdseye.mcts_utils.trim_tree(Q, N, action)
# end_trim = timer() 
# print(f"{Q=}")
# print(f"{N=}")
# print(f"trim: {end_trim-start_trim}")
# end = timer() 
# print(f"time = {end-start}")
# #LightMCTS(env, 4, 1, 1000)