# Configurations Documentation 

1. [Parameter Definitions](#methods-parameters)
2. [Example MCTS config](#example-mcts-config-file)
3. [Example DQN config](#example-dqn-config-file)

___


All configuration files (.yaml) must contain `[Methods]` and `[Defaults]` sections.
``` yaml
[Methods]
...

[Defaults]
...
```
## `[Methods]` Parameters 
``` yaml
[Methods]
method : string
    Choose from ['mcts','dqn']
sensor : string
    Choose from ['drone','signalstrength', 'bearing']
reward : string
    Choose from ['range_reward', 'entropy_collision_reward']
target_speed : float
    Speed of the target
target_start : float
    Starting distance between target and sensor
resample_proportion: float
    Ratio of particles to resample from the prior at every particle filter update
action : simpleactions
state : rfstate 
```

## MCTS specific `[Defaults]` Parameters 
``` yaml
[Defaults]
simulations : int
    Number of simulations
depth : int
    Tree depth
lambda_arg : float
    Lambda value
trials : int
    Number of runs
iterations : int
    Number of iterations
collision : float
    Reward value for collision
loss : float
    Reward value for loss function
plotting : bool
    Flag to plot or not
```

## DQN specific `[Defualts]` parameters 
``` yaml
[Defaults]
log_path : string
    Path for logging output
use_gpu : bool
    If True use GPU device
number_timesteps : int
    Number of timesteps
dueling : bool
    If True dueling value estimation will be used
save_path : string
    Path for saving
save_interval : int
    Interval for saving output values
ob_scale : int
    Scale for observation
gamma : float
    Gamma input value
grad_norm : float
    Max norm value of the gradients to be used in gradient clipping
double_q  : bool
    If True double DQN will be used
param_noise : bool
    If True use parameter space noise
exploration_fraction : float
    Fraction of entire training period over which the exploration rate is annealed
exploration_final_eps : float
    Final value of random action probability
batch_size : int
    Size of a batched sampled from replay buffer for training
train_freq : int
    Update the model every `train_freq` steps
learning_starts : int
    How many steps of the model to collect transitions for before learning starts
target_network_update_freq : int
    Update the target network every `target_network_update_freq` steps
buffer_size : int
    Size of the replay buffer
prioritized_replay : bool
    If True prioritized replay buffer will be used.
prioritized_replay_alpha : float
    Alpha parameter for prioritized replay
prioritized_replay_beta0 : float
    Beta parameter for prioritized replay
atom_num : int
    Atom number in distributional RL for atom_num > 1
min_value : float
    Min value in distributional RL
max_value : float
    Max value in distributional RL
plotting : bool
    Flag to plot or not
```

## Example [MCTS config file](configs/mcts.yaml)
``` yaml
[Methods]
method : mcts
action : simpleactions
sensor : drone
state : rfstate 
target_speed: 1.0
target_start: 100
resample_proportion: 0.005
reward: range_reward

[Defaults]
lambda_arg : 0.8
collision : -2.
loss : -2.
depth : 10
simulations : 200
plotting : False
trials : 500
iterations : 150
```

## Example [DQN config file](configs/dqn.yaml) 
``` yaml
[Methods]
method : dqn
action : simpleactions
sensor : drone
state : rfstate 
target_speed: 1.0
target_start: 100
resample_proportion: 0.1
reward: entropy_collision_reward

[Defaults]
number_timesteps : 100000
dueling : True
double_q : True
param_noise : True
exploration_fraction : 0.2
exploration_final_eps : 0.1
batch_size : 64
train_freq : 4
learning_starts : 100
target_network_update_freq : 600
buffer_size : 10000
prioritized_replay : True
prioritized_replay_alpha : 0.6
prioritized_replay_beta0 : 0.4
min_value : -10
max_value : 10
max_episode_length : 150
atom_num : 1
ob_scale : 1
gamma : 0.99
grad_norm : 10.0
save_interval : 100000
save_path : checkpoints
log_path : rl_log
use_gpu : True
plotting : False
```