import torch
import torch.nn as nn

import numpy as np

class NoisyFCLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        mu_init_range = 1.0/np.sqrt(float(input_dim))
        sig_init_val = 0.5*mu_init_range
        
        w_mu = torch.empty(input_dim, output_dim)
        nn.init.uniform_(w_mu,-mu_init_range,mu_init_range)
        self.w_mu = nn.Parameter(w_mu)
        w_sig = torch.empty(input_dim, output_dim)
        nn.init.constant_(w_sig, sig_init_val)
        self.w_sig = nn.Parameter(w_sig)
        
        b_mu = torch.empty(output_dim)
        nn.init.uniform_(b_mu, -mu_init_range, mu_init_range)
        self.b_mu = nn.Parameter(b_mu)
        b_sig = torch.empty(output_dim)
        nn.init.constant_(b_sig, sig_init_val)
        self.b_sig = nn.Parameter(b_sig)

        self.w_ep = None
        self.b_ep = None
        self.sample_noise()

    def forward(self, x):
        weight = self.w_mu + torch.mul(self.w_sig, self.w_ep)
        bias = self.b_mu + torch.mul(self.b_sig, self.b_ep)
        return torch.matmul(x, weight) + bias

    def sample_noise(self):
        w_ep1 = torch.normal(0.0, 1.0, size=(self.input_dim,))
        w_ep1 = torch.sign(w_ep1)*torch.sqrt(torch.abs(w_ep1))
        w_ep2 = torch.normal(0.0, 1.0, size=(self.output_dim,))
        w_ep2 = torch.sign(w_ep2)*torch.sqrt(torch.abs(w_ep2))
        self.w_ep = torch.matmul(w_ep1.view(-1,1),w_ep2.view(1,-1))
        self.w_ep = self.w_ep.to(self.device)
        
        b_ep = torch.normal(0.0, 1.0, size=(self.output_dim,))
        self.b_ep = torch.sign(b_ep)*torch.sqrt(torch.abs(b_ep))
        self.b_ep = self.b_ep.to(self.device)

