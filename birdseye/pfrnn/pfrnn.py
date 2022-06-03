import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from birdseye.pfrnn.model import Localizer
from birdseye.utils import pol2cart

def parse_args(arg_string=None):

    parser = argparse.ArgumentParser()

    #parser.add('-c', '--config', required=True, default='./configs/train.conf',
    #           is_config_file=True, help='load the config file')

    parser.add_argument('--epochs', type=int, default=800, help='num epochs')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--h', type=int, default=64, help='hidden dim of lstm')
    parser.add_argument('--emb_map', type=int, default=64,
                        help='map embedding dim')
    parser.add_argument('--emb_obs', type=int, default=32,
                        help='observation embedding dim')
    parser.add_argument('--emb_act', type=int, default=32,
                        help='action embedding dim')
    parser.add_argument('--ext_obs', type=int, default=32,
                        help='the size of o(x) in PF-RNNs')
    parser.add_argument('--ext_act', type=int, default=32,
                        help='the size of u(x) in PF-RNNs')

    parser.add_argument('--dropout', type=float,
                        default=0.5, help='dropout rate')
    parser.add_argument('--optim', type=str, default='RMSProp',
                        help='type of optim')
    parser.add_argument('--num_particles', type=int,
                        default=30, help='num of particles')
    parser.add_argument('--sl', type=int, default=100, help='sequence length')
    parser.add_argument('--num_trajs', type=int, default=10000,
                        help='number of trajs')
    parser.add_argument('--resamp_alpha', type=float,
                        default=0.5, help='the soft resampling ratio')
    parser.add_argument('--clip', type=float, default=3.0,
                        help='the grad clip value')
    parser.add_argument('--bp_length', type=int, default=10,
                        help='the truncated bptt length')
    parser.add_argument('--mode', type=str,
                        default='train', help='train or eval')
    parser.add_argument('--model', type=str, default='PFLSTM',
                        help='which model to use for training')
    parser.add_argument('--map_size', type=int, default=10, help='map size')
    parser.add_argument('--act_size', type=int, default=6, help='action space size')
    parser.add_argument('--gpu', type=bool, default=True,
                        help='whether to use GPU')
    parser.add_argument('--bpdecay', type=float, default=0.1,
                        help='the decay along seq for pfrnns')
    parser.add_argument('--obs_num', type=int, default=1, help='observation num')
    parser.add_argument('--h_weight', type=float, default=0.1, help='weight for heading loss')
    parser.add_argument('--l2_weight', type=float, default=1.0, help='weight for l2 loss')
    parser.add_argument('--l1_weight', type=float, default=0.0, help='weight for l1 loss')
    parser.add_argument('--elbo_weight', type=float, default=1.0, help='weight for ELBO loss')

    parser.add_argument('--logs_num', type=int, default=0, help='number of logs folder for your trained model')

    if arg_string is not None:
        args = parser.parse_args(arg_string)
    else:
        args = parser.parse_args()
    return args


def get_optim(args, model):
    if args.optim == 'RMSProp':
        optim = torch.optim.RMSprop(
            model.parameters(), lr=args.lr)
    elif args.optim == 'Adam':
        optim = torch.optim.Adam(
            model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    return optim



class pfrnn:
    def __init__(self):
        self.args = parse_args(arg_string=[])
        self.model = Localizer(self.args)
        self.optimizer = get_optim(self.args, self.model)
        self.particles = None

    def update(self, observation, absolute_pos, action_index):
        # env_map: [batch, *occupancy array (2D)]
        # obs: sensor observation [batch, sequence, *array]
        # pos: target position [batch, sequence, x, y, theta (radians)]
        # action: [batch, sequence, *one hot vector]

        env_map = torch.tensor(np.zeros((self.args.map_size,self.args.map_size))).view(1,1,self.args.map_size, self.args.map_size).float()

        obs = torch.tensor(observation).view(1,1,-1).float()

        x, y = pol2cart(absolute_pos[0], absolute_pos[1])
        pos = torch.tensor([x, y, np.radians(absolute_pos[2])]).view(1,1,-1).float()

        action = torch.tensor(np.zeros(self.args.act_size)).float()
        action[action_index] = 1
        action = action.view(1,1,-1)


        if torch.cuda.is_available() and self.args.gpu:
            env_map = env_map.to('cuda')
            obs = obs.to('cuda')
            pos = pos.to('cuda')
            action = action.to('cuda')

        self.model.zero_grad()
        loss, log_loss, particle_pred = self.model.step(
            env_map, obs, action, pos, self.args)
        self.plot_particles(particle_pred.detach().numpy())
        loss.backward()
        if self.args.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.optimizer.step()

        return particle_pred

    def prep_data(self, observation, absolute_pos, action_index ):
        env_map = torch.tensor(np.zeros((self.args.map_size,self.args.map_size))).view(1,1,self.args.map_size, self.args.map_size).float()

        obs = torch.tensor(observation).view(1,1,-1).float()

        x, y = pol2cart(absolute_pos[0], absolute_pos[1])
        pos = torch.tensor([x, y, np.radians(absolute_pos[2])]).view(1,1,-1).float()

        action = torch.tensor(np.zeros(self.args.act_size)).float()
        action[action_index] = 1
        action = action.view(1,1,-1)

        return (env_map, obs, pos, action)

    def plot_particles(self, particles):
        plt.figure()
        plt.plot(particles[:,0,0,0], particles[:,0,0,1], 'ro')
        plt.show()
