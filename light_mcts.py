import birdseye.utils
import birdseye.sensor
import birdseye.actions
import birdseye.state
import birdseye.env
import birdseye.mcts_utils
from sys import getsizeof
import numpy as np 


class LightMCTS:
    def __init__(self, env, depth, c, simulations=1000):

        self.env = env
        self.depth = depth
        self.c = c
        self.simulations = simulations

        self.Q = {}
        self.N = {}

        self.action = None

    def run(self, belief):

        self.env.reset()

        if self.action is not None:
            self.Q = {}
            self.N = {}

        self.Q, self.N, self.action = birdseye.mcts_utils.select_action_light(
            self.env,
            self.Q,
            self.N,
            self.depth,
            self.c,
            self.simulations,
        )

        return self.action

n_targets = 2

antenna_filename = "radiation_pattern_yagi_5.csv"
power_tx = [26,26]
directivity_tx = [1,1]
freq = [5.7e9, 5.7e9]
fading_sigma = 8
target_speed = 0.5
sensor_speed = 1.0
reward_func = lambda pf: pf.weight_entropy

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
    sensor=sensor, actions=actions, state=state, simulated=True, num_particles=500
)
env.reset()

print(actions.get_action_list())
print(actions.avail_actions())

new_pf = env.pf_copy()

a = birdseye.mcts_utils.select_action_light(env)
print(a)
LightMCTS(env, 4, 1, 1000)