from birdseye.mcts import * 
from birdseye.actions import * 
from birdseye.sensor import * 
from birdseye.state import * 
from birdseye.env import RFEnv 

N = 500
depth = 10
lambda_arg = 0.8 
num_runs = 100 
iterations = 2000
collision = -2
loss = -2

actions = SimpleActions()
sensor = Drone()
state = RFState()
env = RFEnv(sensor, actions, state)

run_mcts(env, N, depth, lambda_arg, num_runs, iterations, collision, loss, plotting=False)
