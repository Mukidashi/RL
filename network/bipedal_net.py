import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


class BipedalTwinQNet(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()

        isize = state_dim[0] + action_dim[0]
        
        self.fc1_1 = nn.Linear(isize, 256)
        self.fc2_1 = nn.Linear(256, 256)
        self.fc3_1 = nn.Linear(256, 1)
        self.fc1_2 = nn.Linear(isize, 256)
        self.fc2_2 = nn.Linear(256, 256)
        self.fc3_2 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat((state,action),1)
        x1 = F.relu(self.fc1_1(x))
        x1 = F.relu(self.fc2_1(x1))
        x2 = F.relu(self.fc1_2(x))
        x2 = F.relu(self.fc2_2(x2))
        return self.fc3_1(x1), self.fc3_2(x2)

    def update_params(self, online_state_dict, tau):
        new_dict = {}
        for key in online_state_dict.keys():
            update_weight = tau*online_state_dict[key] + (1.0-tau)*self.state_dict()[key]
            new_dict[key] = update_weight    
        self.load_state_dict(new_dict)


class BipedalGaussianPolicyNet(nn.Module):

    def __init__(self, state_dim, action_dim, action_bound=None):
        super().__init__()

        state_dim = state_dim[0]
        action_dim = action_dim[0]
        self.action_dim = action_dim

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        bound_mask = np.zeros(self.action_dim)
        if action_bound is not None:
            self.low_bound = action_bound[0]
            self.high_bound = action_bound[1]
            for i, (l,h) in enumerate(zip(self.low_bound,self.high_bound)):
                if l == float('inf') or l == -float('inf') or h == float('inf') or h == -float('inf'):
                    self.low_bound[i] = 0.0
                    self.high_bound[i] = 0.0
                else:
                    bound_mask[i] = 1.0
        else:
            self.low_bound = np.zeros(self.action_dim)
            self.high_bound = np.zeros(self.action_dim)
        self.bound_mask = torch.FloatTensor(bound_mask).view(1,-1).to(self.device)
        self.low_bound = torch.FloatTensor(self.low_bound).view(1,-1).to(self.device)
        self.high_bound = torch.FloatTensor(self.high_bound).view(1,-1).to(self.device)

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2*action_dim)

        self.tanh_fn1 = nn.Tanh()
        self.tanh_fn2 = nn.Tanh()


    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        mu = self.tanh_fn1(x[:,:self.action_dim])
        sig = torch.exp(x[:,self.action_dim:])

        ep = torch.normal(0.0, 1.0, size=mu.shape).to(self.device)
        raw_action = mu + ep*sig
        raw_action_tanh = self.tanh_fn2(raw_action)
        action = (raw_action_tanh + 1.0)*(self.high_bound-self.low_bound)*0.5 + self.low_bound
        action = self.bound_mask*action + (1.0-self.bound_mask)*raw_action

        logprob = -torch.square((raw_action-mu)/sig)/2.0 - torch.log(sig) - 0.5*np.log(2.0*np.pi)
        logprob_correction = - torch.log(1.0-torch.square(raw_action_tanh) + 1.0e-6)
        logprob = logprob + self.bound_mask*logprob_correction
        logprob = torch.sum(logprob,1)

        return action, logprob