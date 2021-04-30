from datetime import datetime
import sys
import configparser
import argparse
import pandas as pd
import os.path
from .mcts_utils import *
from .actions import *
from .sensor import *
from .state import *
from .definitions import *
from .env import RFEnv
from .utils import write_header_log

# Default MCTS inputs
mcts_defaults = {
    'lambda_arg' : 0.8,
    'collision' : -2.,
    'loss' : -2.,
    'depth' : 10,
    'simulations' : 500,
    'plotting' : False,
    'trials' : 100,
    'iterations' : 2000
}


def run_mcts(env, config=None, fig=None, ax=None, global_start_time=None):
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
    num_runs : int
        Number of runs
    iterations : int
        Number of iterations
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
        config = mcts_defaults
    simulations = config.simulations
    DEPTH = config.depth
    lambda_arg = config.lambda_arg
    num_runs = config.trials
    iterations = config.iterations
    COLLISION_REWARD = config.collision
    LOSS_REWARD = config.loss
    plotting = config.plotting

    #cumulative collisions, losses, and number of trials
    #total reward, and best average tracking
    cum_coll = 0
    cum_loss = 0
    cum_trials = 0
    total_reward = 0
    best_average = 1

    run_data = []

    # trials
    mcts_loss = 0
    mcts_coll = 0
    run_times = []
    for i  in range(1, num_runs+1):
        run_start_time = datetime.now()
        #global mcts_loss, mcts_coll, num_particles, DEPTH
        result = mcts_trial(env, iterations, DEPTH, 20, plotting, simulations, fig=fig, ax=ax)
        run_time = datetime.now()-run_start_time
        run_times.append(run_time)
        mcts_coll += result[2]
        mcts_loss += result[3]
        run_data.append([datetime.now(), run_time]+result[1:])
       
        print(".")
        print("\n==============================")
        print("Runs: {}".format(i))
        print("NUM PARTICLES: {}".format(simulations))
        print("MCTS Depth {} Results".format(DEPTH))
        print("Collision Rate: {}".format(mcts_coll/i))
        print("Loss Rate: {}".format(mcts_loss/i))
        print("==============================")
        

        namefile = '{}/mcts/{}_data.csv'.format(RUN_DIR, global_start_time)
        df = pd.DataFrame(run_data, columns=['time','run_time','total_reward','total_col','total_lost', 'avg_r_err', 'avg_theta_err', 'avg_heading_err', 'avg_centroid_err', 'average_rmse'])
        df.to_csv(namefile)


def mcts(args=None, env=None):
    # Grab mcts specific defaults
    defaults = mcts_defaults

    if args:
        config = configparser.ConfigParser(defaults)
        config.read_dict({section: dict(args[section]) for section in args.sections()})
        defaults = dict(config.items('Defaults'))
        # Fix for boolean args
        defaults['plotting'] = config.getboolean('Defaults', 'plotting')
    
    parser = argparse.ArgumentParser(description='Monte Carlo Tree Search',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_defaults(**defaults)
    parser.add_argument('--lambda_arg', type=float, help='Lambda value')
    parser.add_argument('--collision', type=float, help='Reward value for collision')
    parser.add_argument('--loss', type=float, help='Reward value for loss function')
    parser.add_argument('--depth', type=float, help='Tree depth')
    parser.add_argument('--simulations', type=int, help='Number of simulations')
    parser.add_argument('--plotting', type=bool, help='Flag to plot or not')
    parser.add_argument('--trials', type=int, help='Number of runs')
    parser.add_argument('--iterations', type=int, help='Number of iterations')
    args,_ = parser.parse_known_args()
    
    if not env:
        # Setup environment
        actions = SimpleActions()
        sensor = Drone()
        state = RFState()
        env = RFEnv(sensor, actions, state)

    global_start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    write_header_log(config, 'mcts', global_start_time)

    run_mcts(env=env, config=args, global_start_time=global_start_time)


if __name__ == '__main__':
    mcts(args=sys.argv[1:])
