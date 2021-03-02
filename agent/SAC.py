import torch
import numpy as np
import os

from memory import ReplayMemory
from network import BipedalTwinQNet, BipedalGaussianPolicyNet

def q_loss(x,y):
    loss = (x-y)**2
    return 0.5*torch.mean(loss)

class SAC:

    def __init__(self, state_dim, action_dim, save_dir, action_bound=None, memory_size=1000000, 
                 batch_size=256, proc_type="train"):

        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_size, batch_size=self.batch_size)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.online_qnet = BipedalTwinQNet(state_dim, action_dim).to(self.device)
        self.target_qnet = BipedalTwinQNet(state_dim, action_dim).to(self.device)
        self.target_qnet.load_state_dict(self.online_qnet.state_dict())
        self.opt_q = torch.optim.Adam(self.online_qnet.parameters(), lr=0.0003)
        self.qloss_fn = q_loss

        self.policy_net = BipedalGaussianPolicyNet(state_dim, action_dim).to(self.device)
        self.opt_pol = torch.optim.Adam(self.policy_net.parameters(), lr=0.0003)

        self.log_alpha = torch.tensor([0.0],dtype=torch.float32,device=self.device,requires_grad=True)
        self.opt_alp = torch.optim.Adam([self.log_alpha], lr=0.0003)


        self.exploration_rate = -1
        self.gamma = 0.99
        self.tau = 0.005
        self.target_entropy = -action_dim[0]

        self.cur_step = 0
        self.learn_every = 1
        self.mem_size_min = memory_size/2

        self.save_every = 5e5
        self.save_dir = save_dir


    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        state = state.unsqueeze(0)
        action, logprob = self.policy_net(state)
        action = action.squeeze()

        self.cur_step += 1

        return action.to('cpu').detach().numpy()


    def memorize(self, state, next_state, action, reward, done):
        
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action)
        reward = torch.DoubleTensor([reward])
        done = torch.BoolTensor([done])

        self.memory.append((state, next_state, action, reward, done))


    def recall(self):

        batch = self.memory.sample()
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))

        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


    def update(self, state, next_state, action, reward, done):

        alpha = torch.exp(self.log_alpha)

        #optimize Qnet
        q1_est, q2_est = self.online_qnet(state, action)
        with torch.no_grad():
            next_action, next_logprob = self.policy_net(next_state)
            q1_tgt, q2_tgt = self.target_qnet(next_state, next_action)
            q_tgt = torch.minimum(q1_tgt, q2_tgt).squeeze()
            next_logprob = torch.sum(next_logprob,1)
            td1_tgt = reward + (1.0 - done.float())*self.gamma*(q_tgt - alpha*next_logprob)
            td2_tgt = reward + (1.0 - done.float())*self.gamma*(q_tgt - alpha*next_logprob)
        
        loss_q = 0.5*(self.qloss_fn(q1_est.squeeze(), td1_tgt) + self.qloss_fn(q2_est.squeeze(), td2_tgt))

        self.opt_q.zero_grad()
        loss_q.backward(retain_graph=True)
        self.opt_q.step()

        #optimize policy net
        action_est, logprob = self.policy_net(next_state)
        logprob = torch.sum(logprob,1)
        q1_est, q2_est = self.online_qnet(state, action_est)
        q_est = torch.minimum(q1_est, q2_est).squeeze()
        loss_p = torch.mean(alpha*logprob - q_est)

        self.opt_pol.zero_grad()
        loss_p.backward(retain_graph=True)
        self.opt_pol.step()

        #optimize alpha
        with torch.no_grad():
            entropy_diff = - logprob - self.target_entropy
        loss_a = torch.mean(alpha*entropy_diff)
        self.opt_alp.zero_grad()
        loss_a.backward()
        self.opt_alp.step()

        #update target Q
        self.target_qnet.update_params(self.online_qnet.state_dict(), self.tau)

        loss = loss_q + loss_p + loss_a
        return loss.item(), -1.0


    def eval(self, state, next_state, action, reward, done, is_end):
        pass


    def observe(self, state, next_state, action, reward, done, is_end):

        self.memorize(state, next_state, action, reward, done)

        if self.cur_step % self.save_every == 0:
            self.save()
        
        if self.memory.get_size() < self.mem_size_min:
            return None, None
        
        if self.cur_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()


        loss_val, gomi = self.update(state, next_state, action, reward, done)

        return loss_val,gomi


    def save(self):
        save_path = os.path.join(self.save_dir,"SAC_{0}.chkpt".format(int(self.cur_step//self.save_every)))

        torch.save(
            dict(
                online_qnet=self.online_qnet.state_dict(),
                target_qnet=self.target_qnet.state_dict(),
                policy_net=self.policy_net.state_dict()                
            ),
            save_path
        )
        print("SACNet saved to {0} at step {1}".format(save_path,self.cur_step))


    def load(self, load_path):
        if not os.path.exists(load_path):
            raise ValueError(f"{load_path} not Exist")

        ckp = torch.load(load_path, map_location=self.device)
        online_qnet = ckp.get('online_qnet')
        target_qnet = ckp.get('target_qnet')
        policy_net = ckp.get('policy_net')

        self.online_qnet.load_state_dict(online_qnet)
        self.target_qnet.load_state_dict(target_qnet)
        self.policy_net.load_state_dict(policy_net)

        print(f"Loading Model at {load_path}")