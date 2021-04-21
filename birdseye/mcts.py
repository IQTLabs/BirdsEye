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


def run_mcts(env, config=None, fig=None, ax=None, full_config=None):
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
    global_start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if full_config is not None: 
        config2log = {section: dict(full_config[section]) for section in full_config.sections()}
    else: 
        config2log = vars(config)

    #output header file
    header_string = ('config: {}\n' +
                     'MCTS Run: {}\n' +
                     'Depth: {}\n' +
                     'N: {}\n' +
                     'Lambda: {}\n' +
                     'Iterations: {}\n' +
                     'Collision Reward: {}\n' +
                     'Loss Reward: {}\n').format(config2log, global_start_time, 
                                                 DEPTH, simulations, lambda_arg,
                                                 iterations, COLLISION_REWARD, LOSS_REWARD)

    #write output header
    run_dir = RUN_DIR
    if not os.path.isdir(RUN_DIR+'/mcts/'): 
        os.mkdir(RUN_DIR+'/mcts/')
    header_filename = "{}/mcts/{}_header.txt".format(RUN_DIR, global_start_time)
    with open(header_filename, "w") as file:
        file.write(header_string)

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
        mcts_coll += result[2]
        mcts_loss += result[3]
        run_data.append(result[1:])
        
        run_times.append(datetime.now()-run_start_time)
        print(".")
        ##if i % 5 == 0:

        print("\n==============================")
        print("Runs: {}".format(i))
        print("NUM PARTICLES: {}".format(simulations))
        print("MCTS Depth {} Results".format(DEPTH))
        print("Collision Rate: {}".format(mcts_coll/i))
        print("Loss Rate: {}".format(mcts_loss/i))
        print("==============================")
        updated_header = header_string + '\nAverage Runtime: {}'.format(np.mean(run_times))
        with open(header_filename, "w") as file:
            file.write(updated_header)

        namefile = '{}/mcts/{}_data.csv'.format(RUN_DIR, global_start_time)
        df = pd.DataFrame(run_data, columns=['total_reward','total_col','total_lost', 'avg_r_err', 'avg_theta_err', 'avg_heading_err', 'avg_centroid_err', 'average_rmse'])
        df.to_csv(namefile)


def mcts(args=None, env=None):
    # Configuration file parser
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument('-c', '--config',
                             help='Specify a configuration file',
                             metavar='FILE')
    args, remaining_argv = conf_parser.parse_known_args()

    # Grab mcts specific defaults
    defaults = mcts_defaults

    if args.config:
        config = configparser.ConfigParser(defaults)
        config.read([args.config])
        defaults = dict(config.items('Defaults'))
        # Fix for boolean args
        defaults['plotting'] = config.getboolean('Defaults', 'plotting')
    
    parser = argparse.ArgumentParser(description='Monte Carlo Tree Search',
                                     parents=[conf_parser],
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
    args = parser.parse_args(remaining_argv)
    
    if not env:
        # Setup environment
        actions = SimpleActions()
        sensor = Drone()
        state = RFState()
        env = RFEnv(sensor, actions, state)

    run_mcts(env=env, config=args, full_config=config)


if __name__ == '__main__':
    mcts(args=sys.argv[1:])
