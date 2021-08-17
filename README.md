# BirdsEye
### Localizing Radio Frequency Targets Using Reinforcement Learning

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
A Docker file has also been provided for ease of use. To run with Docker, execute the following commands:
```
> docker build -t BirdsEye .
> docker run -it --ipc=host --gpus all BirdsEye -c {config.yaml}
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

    
