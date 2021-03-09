from datetime import datetime
import argparse
import pandas as pd
import os.path
from .mcts_utils import *
from .actions import *
from .sensor import *
from .definitions import *
from birdseye.sensor import * 




def run_mcts(actions, sensor, N, DEPTH, lambda_arg, num_runs, iterations, COLLISION_REWARD, LOSS_REWARD, plotting, fig=None, ax=None):

    global_start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    plotting = plotting
    testing = False
    #epochsize = 500

    #output header file
    header_string = ('MCTS Run: {}\n' +
                     'Depth: {}\n' +
                     'N: {}\n' +
                     'Lambda: {}\n' +
                     'Iterations: {}\n' +
                     'Collision Reward: {}\n' +
                     'Loss Reward: {}\n').format(global_start_time, DEPTH, N, lambda_arg, iterations, COLLISION_REWARD, LOSS_REWARD)

    #write output header
    run_dir = RUN_DIR
    if not os.path.isdir(RUN_DIR): 
        os.mkdir(RUN_DIR)
    header_filename = "{}/{}_header.txt".format(RUN_DIR, global_start_time)
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
    num_particles = N
    mcts_loss = 0
    mcts_coll = 0
    run_times = []
    for i  in range(num_runs):
        run_start_time = datetime.now()
        #global mcts_loss, mcts_coll, num_particles, DEPTH
        result = mcts_trial(actions, sensor, DEPTH, 20, plotting, num_particles, fig=fig, ax=ax)
        mcts_coll += result[2]
        mcts_loss += result[3]
        run_data.append(result[2:4])
        print(".")
        run_times.append(datetime.now()-run_start_time)
        if i % 5 == 0:

            print("\n==============================")
            print("Trials: "+ i)
            print("NUM PARTICLES: "+ num_particles)
            print("MCTS Depth "+ DEPTH+ " Results")
            print("Collision Rate: "+ mcts_coll/i)
            print("Loss Rate: "+ mcts_loss/i)
            print("==============================")
            namefile = '{}_data.csv'.format(global_start_time)
            updated_header = header_string + '\nAverage Runtime: {}'.format(np.mean(run_times))

            with open(header_filename, "w") as file:
                file.write(updated_header)

            df = pd.DataFrame(run_data)
            df.to_csv(namefile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Monte Carlo Tree Search')
    parser.add_argument('--lambda', type=float, default=0.8, dest='lambda_arg')
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--collision', type=float, default=-2)
    parser.add_argument('--loss', type=float, default=-2)
    parser.add_argument('--depth', type=float, default=10)
    parser.add_argument('--N', type=int, default=500)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--iterations', type=float, default=2000)
    parser.add_argument('--plot_header', type=str, default='out')
    args = parser.parse_args()

    actions = SimpleActions()
    sensor = Drone() 
    run_mcts(actions, sensor, args.N, args.depth, args.lambda_arg, args.trials, args.iterations, args.collision, args.loss, False)
