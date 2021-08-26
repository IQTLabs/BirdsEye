# BirdsEye
### Localizing Radio Frequency Targets Using Reinforcement Learning

This repository provides code for simulating the tracking of mobile radio frequency targets using a mobile sensor. The target state estimation is performed using a particle filter and the sensor's movement is controlled through various machine learning methods. We have included code for training and evaluating Monte Carlo Tree Search and Deep Q-Network models. Tracking visualization, localization error and belief uncertainty metrics are also reported. 

![particles](data/example.gif)
## Installation 

```
pip install -r requirements.txt
```

## Usage
### To run on commandline
```
$ python run_birdseye.py -h 
usage: run_birdseye.py [-h] -c FILE [-b]

optional arguments:
  -h, --help            show this help message and exit
  -c FILE, --config FILE
                        Specify a configuration file
  -b, --batch           Perform batch run
```

### To run using a Docker container
First install Docker with GPU support. [Instructions here.](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

A Docker file has also been provided for ease of use. To run with Docker, execute the following commands:
```
> docker build -t birds_eye .
> docker run -it --gpus all birds_eye -c {config.yaml}
```
In order to streamline this process a `Makefile` has been provided as a shorthand. 
```
> make run_mcts
> make run_dqn
> make run_batch
```
Accepted make values are: `run_mcts, run_dqn, run_batch, build`


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

## Description
All code for training and evaluation in simulation is contained in the [birdseye](birdseye) directory.  
The [birdseye](birdseye) directory contains some important base classes which can be extended to offer customizability to a specific use case. We provide a few subclasses to get started. 

### [Sensor](birdseye/sensor.py)
The Sensor class defines the observation model for the simulated sensor. Users must define methods for sampling an observation given a state and for determining the likelihood of an observation given a state. We have provided example subclasses for an omni-directional signal strength setup and a bearing based directional setup. 

### [Actions](birdseye/actions.py)
The Actions class defines the action space for the sensor. For computational simplicity, actions are discretized. 

### [State](birdseye/state.py)
The State class includes methods for updating the state variables of the environment. This includes states for both the sensor and target. Motion dynamics and reward functions are defined within this class. We have included example reward functions based on an entropy/collision tradeoff as well as a range based reward. 

### [RFEnv](birdseye/env.py)
The RFEnv class is a Gym-like class for controlling the entire pipeline of the simulation. 
