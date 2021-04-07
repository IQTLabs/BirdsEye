"""
These functions are adapted from github.com/Officium/RL-Experiments

"""
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
from .rl_common.logger import init_logger
from .rl_common.models import CNN, MLP

from .actions import *
from .sensor import *
from .state import *
from .definitions import *
from .env import RFEnv


def run_dqn(env,
          log_path, use_gpu,
          number_timesteps, dueling,
          save_path, save_interval, ob_scale,
          gamma, grad_norm, double_q, param_noise,
          exploration_fraction, exploration_final_eps,
          batch_size, train_freq, learning_starts, target_network_update_freq,
          buffer_size, prioritized_replay, prioritized_replay_alpha,
          prioritized_replay_beta0, atom_num, min_value, max_value, max_episode_length):
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
   
    # Setup logging
    logger = init_logger(log_path)

    # Access requested device
    device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')

    # Define network & training optimizer
    policy_dim = len(env.actions.action_space)
    in_dim = (1, 100, 100)
    network = CNN(in_dim, policy_dim, atom_num, dueling)
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
                if extra:
                    loss = (extra[0] * huber_loss(abs_td_error)).mean()
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
                buffer.update_priorities(extra[1], priorities)

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
                       os.path.join(save_path, '{}.checkpoint'.format(n_iter)))


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument('--number_timesteps', type=int, default=10000)
    parser.add_argument('--no_dueling', action='store_false', dest='dueling')
    parser.add_argument('--no_double_q', action='store_false', dest='double_q')
    parser.add_argument('--param_noise', action='store_true')

    parser.add_argument('--exploration_fraction', type=float, default=0.2)
    parser.add_argument('--exploration_final_eps', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--train_freq', type=int, default=4)
    parser.add_argument('--learning_starts', type=int, default=100)
    parser.add_argument('--target_network_update_freq', type=int, default=100)
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--prioritized_replay', action='store_true')
    parser.add_argument('--prioritized_replay_alpha', type=float, default=0.6)
    parser.add_argument('--prioritized_replay_beta0', type=float, default=0.4)
    parser.add_argument('--min_value', type=int, default=-10)
    parser.add_argument('--max_value', type=int, default=10)
    parser.add_argument('--max_episode_length', type=int, default=100)

    parser.add_argument('--atom_num', type=int, default=1)
    parser.add_argument('--ob_scale', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--grad_norm', type=float, default=10.0)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--save_path', type=str, default='checkpoints', help='save model every x steps(0 = disabled)')
    parser.add_argument('--log_path', type=str, default='rl_log')
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    # Setup environment
    actions = SimpleActions()
    sensor = Drone() 
    state = RFState() 
    env = RFEnv(sensor, actions, state)

    # Run dqn method
    run_dqn(env,
            args.log_path, args.use_gpu,
            args.number_timesteps, args.dueling, 
            args.save_path, args.save_interval, args.ob_scale,
            args.gamma, args.grad_norm, args.double_q, args.param_noise,
            args.exploration_fraction, args.exploration_final_eps,
            args.batch_size, args.train_freq, args.learning_starts, args.target_network_update_freq,
            args.buffer_size, args.prioritized_replay, args.prioritized_replay_alpha,
            args.prioritized_replay_beta0, args.atom_num, args.min_value, args.max_value, args.max_episode_length)


