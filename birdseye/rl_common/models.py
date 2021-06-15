"""
These functions are adapted from github.com/Officium/RL-Experiments

"""
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from torch.optim import Adam

from birdseye.rl_common.util import Flatten

class SmallRFPFQnet(nn.Module): 
    def __init__(self, map_dim, state_dim, policy_dim, atom_num, dueling): 
        super().__init__()
        self.atom_num = atom_num
        self.map_dim = map_dim
        self.state_dim = state_dim
        c, h, w = map_dim
        cnn_out_dim = 64 * ((h - 21) // 8) * ((w - 21) // 8)
        self.map_feature = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(cnn_out_dim, 50),
            nn.ReLU(True),
        )

        self.state_feature = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(True), 
            nn.Linear(64, 50),
            nn.ReLU(True),
        )

        self.joint_feature = nn.Sequential(
            nn.Linear(100, 50), 
            nn.ReLU(True), 
        )

        self.q = nn.Sequential(
            nn.Linear(50, policy_dim * atom_num)
        )

        if dueling:
            self.state = nn.Sequential(
                nn.Linear(50, atom_num)
            )

        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        state = x[:,:self.state_dim]
        pf_map = x[:,self.state_dim:].view(x.size(0), self.map_dim[0], self.map_dim[1], self.map_dim[2])
        assert state.size(0) == pf_map.size(0)
        batch_size = state.size(0)
        map_latent = self.map_feature(pf_map)
        state_latent = self.state_feature(state)
        joint_latent = self.joint_feature(torch.cat((state_latent, map_latent), dim=1))
        qvalue = self.q(joint_latent)

        if self.atom_num == 1:
            if hasattr(self, 'state'):
                svalue = self.state(joint_latent)
                qvalue = svalue + qvalue - qvalue.mean(1, keepdim=True)
            return qvalue
        else:
            qvalue = qvalue.view(batch_size, -1, self.atom_num)
            if hasattr(self, 'state'):
                svalue = self.state(joint_latent).unsqueeze(1)
                qvalue = svalue + qvalue - qvalue.mean(1, keepdim=True)
            logprobs = log_softmax(qvalue, -1)
            return logprobs

class RFPFQnet(nn.Module): 
    def __init__(self, map_dim, state_dim, policy_dim, atom_num, dueling): 
        super().__init__()
        self.atom_num = atom_num
        self.map_dim = map_dim
        self.state_dim = state_dim
        c, h, w = map_dim
        cnn_out_dim = 64 * ((h - 28) // 8) * ((w - 28) // 8)
        self.map_feature = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(cnn_out_dim, 100),
            nn.ReLU(True),
        )

        self.state_feature = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(True), 
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 100),
            nn.ReLU(True),
        )

        self.joint_feature = nn.Sequential(
            nn.Linear(200, 100), 
            nn.ReLU(True), 
        )

        self.q = nn.Sequential(
            nn.Linear(100, policy_dim * atom_num)
        )

        if dueling:
            self.state = nn.Sequential(
                nn.Linear(100, atom_num)
            )

        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        state = x[:,:self.state_dim]
        pf_map = x[:,self.state_dim:].view(x.size(0), self.map_dim[0], self.map_dim[1], self.map_dim[2])
        assert state.size(0) == pf_map.size(0)
        batch_size = state.size(0)
        map_latent = self.map_feature(pf_map)
        state_latent = self.state_feature(state)
        joint_latent = self.joint_feature(torch.cat((state_latent, map_latent), dim=1))
        qvalue = self.q(joint_latent)

        if self.atom_num == 1:
            if hasattr(self, 'state'):
                svalue = self.state(joint_latent)
                qvalue = svalue + qvalue - qvalue.mean(1, keepdim=True)
            return qvalue
        else:
            qvalue = qvalue.view(batch_size, -1, self.atom_num)
            if hasattr(self, 'state'):
                svalue = self.state(joint_latent).unsqueeze(1)
                qvalue = svalue + qvalue - qvalue.mean(1, keepdim=True)
            logprobs = log_softmax(qvalue, -1)
            return logprobs

        

class CNN(nn.Module):
    def __init__(self, in_shape, out_dim, atom_num, dueling):
        super().__init__()
        c, h, w = in_shape
        cnn_out_dim = 64 * ((h - 28) // 8) * ((w - 28) // 8)
        self.atom_num = atom_num
        self.feature = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            Flatten(),
        )

        self.q = nn.Sequential(
            nn.Linear(cnn_out_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, out_dim * atom_num)
        )
        if dueling:
            self.state = nn.Sequential(
                nn.Linear(cnn_out_dim, 256),
                nn.ReLU(True),
                nn.Linear(256, atom_num)
            )

        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        latent = self.feature(x)
        qvalue = self.q(latent)
        if self.atom_num == 1:
            if hasattr(self, 'state'):
                svalue = self.state(latent)
                qvalue = svalue + qvalue - qvalue.mean(1, keepdim=True)
            return qvalue
        else:
            qvalue = qvalue.view(batch_size, -1, self.atom_num)
            if hasattr(self, 'state'):
                svalue = self.state(latent).unsqueeze(1)
                qvalue = svalue + qvalue - qvalue.mean(1, keepdim=True)
            logprobs = log_softmax(qvalue, -1)
            return logprobs


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, atom_num, dueling):
        super().__init__()
        self.atom_num = atom_num
        self.feature = nn.Sequential(
            Flatten(),
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.q = nn.Linear(64, out_dim * atom_num)
        if dueling:
            self.state = nn.Linear(64, atom_num)

        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        latent = self.feature(x)
        qvalue = self.q(latent)
        if self.atom_num == 1:
            if hasattr(self, 'state'):
                svalue = self.state(latent)
                qvalue = svalue + qvalue - qvalue.mean(1, keepdim=True)
            return qvalue
        else:
            if hasattr(self, 'state'):
                qvalue = qvalue.view(batch_size, -1, self.atom_num)
                svalue = self.state(latent).unsqueeze(1)
                qvalue = svalue + qvalue - qvalue.mean(1, keepdim=True)
            logprobs = log_softmax(qvalue, -1)
            return logprobs