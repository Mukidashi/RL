import torch
from torch import nn
import torch.nn.functional as F

from .network_util import conv2d_size_out
from layer import NoisyFCLayer


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


class MarioDuelNet(nn.Module):

    def __init__(self, input_dim, action_dim):
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
        self.fc1_a = nn.Linear(lsize,512)
        self.fc2_a = nn.Linear(512,action_dim)
        self.fc1_v = nn.Linear(lsize,512)
        self.fc2_v = nn.Linear(512,1)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        xa = F.relu(self.fc1_a(torch.flatten(x,start_dim=1)))
        xa = self.fc2_a(xa)
        xv = F.relu(self.fc1_v(torch.flatten(x,start_dim=1)))
        xv = self.fc2_v(xv)
        return xv, xa


class MarioNoisyDuelNet(nn.Module):

    def __init__(self, input_dim, action_dim):
        super().__init__()

        c,h,w = input_dim

        self.conv1 = nn.Conv2d(c,32,kernel_size=8,stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        #add
        # self.conv4 = nn.Conv2d(64,64,kernel_size=3, stride=1)
        # self.bn4 = nn.BatchNorm2d(64)

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2),3,1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2),3,1)
        lsize = convw*convh*64
        # convw = conv2d_size_out(convw,3,1)
        # convh = conv2d_size_out(convh,3,1)
        # lsize = convw*convh*64

        self.fc1_a = NoisyFCLayer(lsize, 512)
        self.fc2_a = NoisyFCLayer(512, action_dim)
        self.fc1_v = NoisyFCLayer(lsize, 512)
        self.fc2_v = NoisyFCLayer(512, 1)

        # self.fc1_a = NoisyFCLayer(lsize, 256)
        # self.fc2_a = NoisyFCLayer(256,action_dim)
        # self.fc1_v = NoisyFCLayer(lsize, 256)
        # self.fc2_v = NoisyFCLayer(256, 1)


    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.bn4(self.conv4(x))) #add

        x = torch.flatten(x,start_dim=1)
        xa = F.relu(self.fc1_a(x))
        xa = self.fc2_a(xa)
        xv = F.relu(self.fc1_v(x))
        xv = self.fc2_v(xv)
        return xv, xa


    def sample_noise(self):

        self.fc1_a.sample_noise()
        self.fc2_a.sample_noise()
        self.fc1_v.sample_noise()
        self.fc2_v.sample_noise()


class MarioCategoricalNet(nn.Module):

    def __init__(self, input_dim, action_dim, atom_num):
        super().__init__()

        self.atom_num = atom_num
        self.action_dim = action_dim

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
        self.linear2 = nn.Linear(512,action_dim*atom_num)

        self.softfn = nn.Softmax(dim=2)


    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.linear1(torch.flatten(x,start_dim=1)))
        x = self.linear2(x)
        return self.softfn(x.view(-1,self.action_dim,self.atom_num))