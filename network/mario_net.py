import torch
from torch import nn
import torch.nn.functional as F

from .network_util import conv2d_size_out

class MarioNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        c,h,w = input_dim

        self.conv1 = nn.Conv2d(c,32,kernel_size=8,stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2),3,1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2),3,1)
        lsize = convw*convh*64
        self.linear1 = nn.Linear(lsize,512)
        self.linear2 = nn.Linear(512,output_dim)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.linear1(torch.flatten(x,start_dim=1)))
        return self.linear2(x)
