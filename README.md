# BirdsEye
## Introduction
### Localizing Radio Frequency Targets Using Reinforcement Learning
The BirdsEye project demonstrates the simulated
tracking of radio frequency (RF) signals via
reinforcement learning (RL) techniques implemented
on low-fidelity sensors. 
This permits the in-situ observation-training RL paradigm without the need
for significant compute hardware such as graphical
processing units (GPU).
Instead, these methods can
be run on low-cost, commercial, off-the-shelf
technology, providing capabilities to applications in
which covert or concealed sensors are paramount in
deployment, or where more sensitive sensors do not
function or cannot be installed due to the nature of
the environment.

### Methods
BirdsEye has implemented two statistical methods which drive how the sensor adaptively
tracks an observed target signal: Monte Carlo Tree Search (MCTS) and Deep Q-Learning
(DQN). While each method has advantages over the other, neither requires heavy
compute resources such as a GPU. The MCTS method performs a stochastic search and
selection of the actions available to the sensor, identifying the decision which maximizes
the return on localization rewards. The DQN method is a reinforcement learning algorithm
which can adapt to large decision spaces using neural networks, with major public
successes such as [DeepMind’s AlphaGo](https://deepmind.com/research/case-studies/alphago-the-story-so-far).

![particles](data/example.gif)

## Usage

### Installation 

```
pip install -r requirements.txt
```

### To Run
```
$ python run_birdseye.py -h 
usage: run_birdseye.py [-h] -c FILE [-b]

optional arguments:
  -h, --help            show this help message and exit
  -c FILE, --config FILE
                        Specify a configuration file
  -b, --batch           Perform batch run
```
### Configurations 
See [Configurations Documentation](CONFIGS.md) for more information. 


## Examples
### Run with Monte Carlo Tree Search policy
```
$ python run_birdseye.py -c configs/mcts.yaml 
```
### Run with Deep Q-Network policy 
```
$ python run_birdseye.py -c configs/dqn.yaml 
```

___


![DQN](data/dqn_arch.png)
> Deep Q-Network architecture

    
