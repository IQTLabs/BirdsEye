"""
These functions are adapted from github.com/Officium/RL-Experiments

"""
from datetime import datetime
import configparser
import argparse
import math
import os
import random
import time
from collections import deque
from copy import deepcopy
import numpy as np

import torch
import torch.distributions
import torch.nn as nn
from torch.nn.functional import softmax, log_softmax
from torch.optim import Adam

from .rl_common.util import scale_ob
from .rl_common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .rl_common.logger import init_logger, close_logger
from .rl_common.models import SmallRFPFQnet

from .actions import *
from .sensor import *
from .state import *
from .definitions import *
from .env import RFEnv
from .utils import tracking_error, write_header_log, Results


# Default DQN inputs
dqn_defaults = {
    'number_timesteps' : 10000,
    'dueling' : False,
    'double_q' : False,
    'param_noise' : True,
    'exploration_fraction' : 0.2,
    'exploration_final_eps' : 0.1,
    'batch_size' : 100,
    'train_freq' : 4,
    'learning_starts' : 100,
    'target_network_update_freq' : 100,
    'buffer_size' : 10000,
    'prioritized_replay' : True,
    'prioritized_replay_alpha' : 0.6,
    'prioritized_replay_beta0' : 0.4,
    'min_value' : -10,
    'max_value' : 10,
    'max_episode_length' : 500,
    'atom_num' : 1,
    'ob_scale' : 1,
    'gamma' : 0.99,
    'grad_norm' : 10.0,
    'save_interval' : 100000,
    'eval_interval' : 100000,
    'save_path' : 'checkpoints',
    'log_path' : 'rl_log',
    'use_gpu' : True,
    'plotting': False,
    'eval_mode': False
}

def simple_prep(env, device, checkpoint_filename):
    policy_dim = len(env.actions.action_space)
    map_dim = (env.state.n_targets, 300, 300)
    network = SmallRFPFQnet(env.state.n_targets, map_dim, env.state.state_dim, policy_dim)
    qnet = network.to(device)
    checkpoint = torch.load(checkpoint_filename, map_location=device)
    qnet.load_state_dict(checkpoint[0])

    return qnet

def simple_run(qnet, observation, device):
    with torch.no_grad():
        observation = torch.from_numpy(np.expand_dims(observation, 0).astype(np.float32)).to(device)
        q_values = qnet(observation)
        action = q_values.argmax(1).cpu().numpy()[0]

        return action


def run_dqn(env, config, global_start_time):
    """Function to run DQN

    Publications:
    Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep
    reinforcement learning[J]. Nature, 2015, 518(7540): 529.
    Hessel M, Modayil J, Van Hasselt H, et al. Rainbow: Combining Improvements
    in Deep Reinforcement Learning[J]. 2017.

    Parameters
    ----------
    env : object
        Environment definitions
    config : object
        Config object which must have following:

    log_path : string
        Path for logging output
    use_gpu : bool
        Flag for using GPU device
    number_timesteps : int
        Number of timesteps
    dueling : bool
        Flag: if True dueling value estimation will be used
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
        Flag: if True double DQN will be used
    param_noise : bool
        Flag: whether or not to use parameter space noise
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
        Flag: if True prioritized replay buffer will be used.
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
    """
    log_path = config.log_path
    use_gpu = config.use_gpu
    number_timesteps = config.number_timesteps
    dueling = config.dueling
    save_path = config.save_path
    save_interval = config.save_interval
    eval_interval = config.eval_interval
    ob_scale = config.ob_scale
    gamma = config.gamma
    grad_norm = config.grad_norm
    double_q = config.double_q
    param_noise = config.param_noise
    exploration_fraction = config.exploration_fraction
    exploration_final_eps = config.exploration_final_eps
    batch_size = config.batch_size
    train_freq = config.train_freq
    learning_starts = config.learning_starts
    target_network_update_freq = config.target_network_update_freq
    buffer_size = config.buffer_size
    prioritized_replay = config.prioritized_replay
    prioritized_replay_alpha = config.prioritized_replay_alpha
    prioritized_replay_beta0 = config.prioritized_replay_beta0
    atom_num = config.atom_num
    min_value = config.min_value
    max_value = config.max_value
    max_episode_length = config.max_episode_length
    plotting = config.plotting
    eval_mode = config.eval_mode

    # Results instance for saving results to file
    results = Results(method_name='dqn',
                        global_start_time=global_start_time,
                        num_iters=number_timesteps,
                        plotting=plotting)

    # Setup logging
    logger = init_logger(log_path)

    # Access requested device
    device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')

    # Define network & training optimizer
    policy_dim = len(env.actions.action_space)
    map_dim = (env.state.n_targets, 300, 300) # TODO: modify to match multi target
    #network = CNN(map_dim, policy_dim, atom_num, dueling)
    #  SmallRFPFQnet(n_targets, map_dim, state_dim, policy_dim, atom_num, dueling)
    network = SmallRFPFQnet(env.state.n_targets, map_dim, 4, policy_dim, atom_num, dueling)
    optimizer = Adam(network.parameters(), 1e-4, eps=1e-5)

    qnet = network.to(device)
    qtar = deepcopy(qnet)
    if prioritized_replay:
        buffer = PrioritizedReplayBuffer(buffer_size, device,
                                         prioritized_replay_alpha,
                                         prioritized_replay_beta0)
    else:
        buffer = ReplayBuffer(buffer_size, device)
    generator = _generate(device, env, qnet, ob_scale,
                          number_timesteps, param_noise,
                          exploration_fraction, exploration_final_eps,
                          atom_num, min_value, max_value, max_episode_length)
    if atom_num > 1:
        delta_z = float(max_value - min_value) / (atom_num - 1)
        z_i = torch.linspace(min_value, max_value, atom_num).to(device)

    infos = {'eplenmean': deque(maxlen=100), 'eprewmean': deque(maxlen=100)}

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    start_ts = time.time()

    if eval_mode in ['True','true',True]:
        checkpoint = torch.load('checkpoints/dqn_doublerssi.checkpoint', map_location=device)
        qnet.load_state_dict(checkpoint[0])
        evaluate(env, qnet, max_episode_length, device, ob_scale, results)
        return

    for n_iter in range(1, number_timesteps + 1):
        if prioritized_replay:
            buffer.beta += (1 - prioritized_replay_beta0) / number_timesteps
        *data, info = generator.__next__()
        buffer.add(*data)
        for k, v in info.items():
            infos[k].append(v)

        # update qnet
        if n_iter > learning_starts and n_iter % train_freq == 0:
            b_o, b_a, b_r, b_o_, b_d, *extra = buffer.sample(batch_size)
            b_o.mul_(ob_scale)
            b_o_.mul_(ob_scale)

            if atom_num == 1:
                with torch.no_grad():
                    if double_q:
                        b_a_ = qnet(b_o_).argmax(1).unsqueeze(1)
                        b_q_ = (1 - b_d) * qtar(b_o_).gather(1, b_a_)
                    else:
                        b_q_ = (1 - b_d) * qtar(b_o_).max(1, keepdim=True)[0]
                b_q = qnet(b_o).gather(1, b_a)
                abs_td_error = (b_q - (b_r + gamma * b_q_)).abs()
                priorities = abs_td_error.detach().cpu().clamp(1e-6).numpy()
                if prioritized_replay:
                    loss = (extra[-2] * huber_loss(abs_td_error)).mean()
                else:
                    loss = huber_loss(abs_td_error).mean()
            else:
                with torch.no_grad():
                    b_dist_ = qtar(b_o_).exp()
                    b_a_ = (b_dist_ * z_i).sum(-1).argmax(1)
                    b_tzj = (gamma * (1 - b_d) * z_i[None, :]
                             + b_r).clamp(min_value, max_value)
                    b_i = (b_tzj - min_value) / delta_z
                    b_l = b_i.floor()
                    b_u = b_i.ceil()
                    b_m = torch.zeros(batch_size, atom_num).to(device)
                    temp = b_dist_[torch.arange(batch_size), b_a_, :]
                    b_m.scatter_add_(1, b_l.long(), temp * (b_u - b_i))
                    b_m.scatter_add_(1, b_u.long(), temp * (b_i - b_l))
                b_q = qnet(b_o)[torch.arange(batch_size), b_a.squeeze(1), :]
                kl_error = -(b_q * b_m).sum(1)
                # use kl error as priorities as proposed by Rainbow
                priorities = kl_error.detach().cpu().clamp(1e-6).numpy()
                loss = kl_error.mean()

            optimizer.zero_grad()
            loss.backward()
            if grad_norm is not None:
                nn.utils.clip_grad_norm_(qnet.parameters(), grad_norm)
            optimizer.step()
            if prioritized_replay:
                buffer.update_priorities(extra[-1], priorities)

        # update target net and log
        if n_iter % target_network_update_freq == 0:
            qtar.load_state_dict(qnet.state_dict())
            logger.info('{} Iter {} {}'.format('=' * 10, n_iter, '=' * 10))
            fps = int(n_iter / (time.time() - start_ts))
            logger.info('Total timesteps {} FPS {}'.format(n_iter, fps))
            for k, v in infos.items():
                v = (sum(v) / len(v)) if v else float('nan')
                logger.info('{}: {:.6f}'.format(k, v))
            if n_iter > learning_starts and n_iter % train_freq == 0:
                logger.info('vloss: {:.6f}'.format(loss.item()))

        if save_interval and n_iter % save_interval == 0:
            torch.save([qnet.state_dict(), optimizer.state_dict()],
                       os.path.join(save_path, '{}_{}.checkpoint'.format(global_start_time, n_iter)))

        if eval_interval and n_iter % eval_interval == 0:
            evaluate(env, qnet, max_episode_length, device, ob_scale, results)

    close_logger(logger)

def evaluate(env, qnet, max_episode_length, device, ob_scale, results):

    trials = 500 #500
    run_data = []
    for i in range(trials):
        run_start_time = datetime.now()
        print('test trial {}/{}'.format(i, trials))
        result = test(env, qnet, max_episode_length, device, ob_scale, results)
        run_time = datetime.now()-run_start_time
        if results.plotting:
            results.save_gif(i)
        run_data.append([datetime.now(), run_time] + result)

    # Saving results to CSV file
    results.write_dataframe(run_data=run_data)


def test(env, qnet, number_timesteps, device, ob_scale, results=None):
    """ Perform one test run """

    o = env.reset()

    # Save values for all iterations and episodes
    all_target_states = [None]*number_timesteps
    all_sensor_states = [None]*number_timesteps
    all_actions = [None]*number_timesteps
    all_obs = [None]*number_timesteps
    all_reward = np.zeros(number_timesteps)
    all_col = np.zeros(number_timesteps)
    all_loss = np.zeros(number_timesteps)
    all_r_err = np.zeros((number_timesteps, env.state.n_targets))
    all_theta_err = np.zeros((number_timesteps, env.state.n_targets))
    all_heading_err = np.zeros((number_timesteps, env.state.n_targets))
    all_centroid_err = np.zeros((number_timesteps, env.state.n_targets))
    all_rmse = np.zeros((number_timesteps, env.state.n_targets))
    all_mae = np.zeros((number_timesteps, env.state.n_targets))
    all_inference_times = np.zeros(number_timesteps)
    all_pf_cov = [None]*number_timesteps

    for n in range(number_timesteps):
        with torch.no_grad():
            ob = scale_ob(np.expand_dims(o, 0), device, ob_scale)

            inference_start_time = datetime.now()
            q = qnet(ob)
            a = q.argmax(1).cpu().numpy()[0]
            inference_time = (datetime.now() - inference_start_time).total_seconds()

            # take action in env
            o, r, done, info = env.step(a)

            # error metrics
            r_error, theta_error, heading_error, centroid_distance_error, rmse, mae  = tracking_error(env.state.target_state, env.pf.particles)

            total_col = np.mean([np.mean(env.pf.particles[:,4*t] < 15) for t in range(env.state.n_targets)])
            total_lost = np.mean([np.mean(env.pf.particles[:,4*t] > 150) for t in range(env.state.n_targets)])

            # for target_state in env.state.target_state:
            #     if target_state[0] < 15:
            #         total_col += 1

            #     if target_state[0] > 150:
            #         total_lost += 1

            # Save results to output arrays
            all_target_states[n] = env.state.target_state
            all_sensor_states[n] = env.state.sensor_state
            all_actions[n] = a
            all_obs[n] = info['observation']
            all_r_err[n] = r_error
            all_theta_err[n] = theta_error
            all_heading_err[n] = heading_error
            all_centroid_err[n] = centroid_distance_error
            all_rmse[n] = rmse
            all_mae[n] = mae
            all_reward[n] = r
            all_col[n] = total_col
            all_loss[n] = total_lost
            all_inference_times[n] = inference_time
            all_pf_cov[n] = list(env.pf.cov_state.flatten())

            if results is not None and results.plotting:
                #results.build_plots(env.state.target_state, env.pf.particles, env.state.sensor_state, env.get_absolute_target(), env.get_absolute_particles(), n, None, None)
                results.build_multitarget_plots(env=env, time_step=n, centroid_distance_error=centroid_distance_error, selected_plots=[4])


    return [all_target_states, all_sensor_states, all_actions,
            all_obs, all_reward, all_col, all_loss, all_r_err,
            all_theta_err, all_heading_err, all_centroid_err, all_rmse, all_mae, all_inference_times, all_pf_cov]


def _generate(device, env, qnet, ob_scale,
              number_timesteps, param_noise,
              exploration_fraction, exploration_final_eps,
              atom_num, min_value, max_value, max_episode_length):
    """ Generate training batch sample """
    noise_scale = 1e-2
    action_dim = len(env.actions.action_space)
    explore_steps = number_timesteps * exploration_fraction
    if atom_num > 1:
        vrange = torch.linspace(min_value, max_value, atom_num).to(device)

    o = env.reset()
    infos = dict()
    for n in range(1, number_timesteps + 1):
        epsilon = 1.0 - (1.0 - exploration_final_eps) * n / explore_steps
        epsilon = max(exploration_final_eps, epsilon)

        # sample action
        with torch.no_grad():
            ob = scale_ob(np.expand_dims(o, 0), device, ob_scale)
            q = qnet(ob)
            if atom_num > 1:
                q = (q.exp() * vrange).sum(2)
            if not param_noise:
                if random.random() < epsilon:
                    a = int(random.random() * action_dim)
                else:
                    a = q.argmax(1).cpu().numpy()[0]
            else:
                # see Appendix C of `https://arxiv.org/abs/1706.01905`
                q_dict = deepcopy(qnet.state_dict())
                for _, m in qnet.named_modules():
                    if isinstance(m, nn.Linear):
                        std = torch.empty_like(m.weight).fill_(noise_scale)
                        m.weight.data.add_(torch.normal(0, std).to(device))
                        std = torch.empty_like(m.bias).fill_(noise_scale)
                        m.bias.data.add_(torch.normal(0, std).to(device))
                q_perturb = qnet(ob)
                if atom_num > 1:
                    q_perturb = (q_perturb.exp() * vrange).sum(2)
                kl_perturb = ((log_softmax(q, 1) - log_softmax(q_perturb, 1)) *
                              softmax(q, 1)).sum(-1).mean()
                kl_explore = -math.log(1 - epsilon + epsilon / action_dim)
                if kl_perturb < kl_explore:
                    noise_scale *= 1.01
                else:
                    noise_scale /= 1.01
                qnet.load_state_dict(q_dict)
                if random.random() < epsilon:
                    a = int(random.random() * action_dim)
                else:
                    a = q_perturb.argmax(1).cpu().numpy()[0]

        # take action in env
        o_, r, done, info = env.step(a)
        if info.get('episode'):
            infos = {
                'eplenmean': info['episode']['l'],
                'eprewmean': info['episode']['r'],
            }
        # return data and update observation
        yield (o, [a], [r], o_, [int(done)], infos)
        infos = dict()
        o = o_ if not done else env.reset()

        if info['episode']['l'] > max_episode_length:
            env.reset()


def huber_loss(abs_td_error):
    flag = (abs_td_error < 1).float()
    return flag * abs_td_error.pow(2) * 0.5 + (1 - flag) * (abs_td_error - 0.5)


def dqn(args=None, env=None):
    defaults = dqn_defaults
    config = None

    if args:
        config = configparser.ConfigParser(defaults)
        config.read_dict({section: dict(args[section]) for section in args.sections()})
        defaults = dict(config.items('Defaults'))
        # Fix for boolean args
        defaults['param_noise'] = config.getboolean('Defaults', 'param_noise')
        defaults['dueling'] = config.getboolean('Defaults', 'dueling')
        defaults['double_q'] = config.getboolean('Defaults', 'double_q')
        defaults['prioritized_replay'] = config.getboolean('Defaults', 'prioritized_replay')
        defaults['use_gpu'] = config.getboolean('Defaults', 'use_gpu')
        defaults['eval_mode'] = config.getboolean('Defaults', 'eval_mode')

    parser = argparse.ArgumentParser(description='DQN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_defaults(**defaults)
    parser.add_argument('--number_timesteps', type=int, help='Number of timesteps')
    parser.add_argument('--dueling', type=bool, help='lag: if True dueling value estimation will be used')
    parser.add_argument('--double_q', type=bool, help='Flag: if True double DQN will be used')
    parser.add_argument('--param_noise', type=bool, help='Flag: whether or not to use parameter space noise')

    parser.add_argument('--exploration_fraction', type=float, help='Fraction of entire training period over which the exploration rate is annealed')
    parser.add_argument('--exploration_final_eps', type=float, help='Final value of random action probability')
    parser.add_argument('--batch_size', type=int, help='Size of a batched sampled from replay buffer for training')
    parser.add_argument('--train_freq', type=int, help='Update the model every `train_freq` steps')
    parser.add_argument('--learning_starts', type=int, help='How many steps of the model to collect transitions for before learning starts')
    parser.add_argument('--target_network_update_freq', type=int, help='Update the target network every `target_network_update_freq` steps')
    parser.add_argument('--buffer_size', type=int, help='Size of the replay buffer')
    parser.add_argument('--prioritized_replay', type=bool, help='Flag: if True prioritized replay buffer will be used.')
    parser.add_argument('--prioritized_replay_alpha', type=float, help='Alpha parameter for prioritized replay')
    parser.add_argument('--prioritized_replay_beta0', type=float, help='Beta parameter for prioritized replay')
    parser.add_argument('--min_value', type=int, help='Min value in distributional RL')
    parser.add_argument('--max_value', type=int, help='Max value in distributional RL')
    parser.add_argument('--max_episode_length', type=int, help='Max episode length')

    parser.add_argument('--atom_num', type=int, help='Atom number in distributional RL for atom_num > 1')
    parser.add_argument('--ob_scale', type=int, help='Scale for observation')
    parser.add_argument('--gamma', type=float, help='Gamma input value')
    parser.add_argument('--grad_norm', type=float, help='Max norm value of the gradients to be used in gradient clipping')
    parser.add_argument('--save_interval', type=int, help='Interval for saving output values')
    parser.add_argument('--eval_interval', type=int, help='Interval for evaluating model')
    parser.add_argument('--save_path', type=str, help='Path for saving')
    parser.add_argument('--log_path', type=str, help='Path for logging output')
    parser.add_argument('--use_gpu', type=bool, help='Flag for using GPU device')
    args, _ = parser.parse_known_args()

    if not env:
        # Setup environment
        actions = SimpleActions()
        sensor = Drone()
        state = RFState()
        env = RFEnv(sensor, actions, state)

    global_start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if config:
        write_header_log(config, 'dqn', global_start_time)

    # Run dqn method
    run_dqn(env=env, config=args, global_start_time=global_start_time)


if __name__ == '__main__':
    dqn()
