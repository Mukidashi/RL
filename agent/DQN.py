import torch
import numpy as np
import os

from memory import ReplayMemory
from network import MarioNet

class DQN:

    def __init__(self, state_dim, action_dim, save_dir, memory_size=100000,batch_size=32,loss_type="MSE"):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_size, batch_size=self.batch_size)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
        self.online_net = MarioNet(state_dim,action_dim).to(self.device)
        self.target_net = MarioNet(state_dim,action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=0.00025)

        if loss_type == "SmoothL1":
            self.loss_fn = torch.nn.SmoothL1Loss()
        else:
            self.loss_fn = torch.nn.MSELoss()

        
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.learn_every = 3
        self.sync_every = 1e4
        self.mem_size_min = memory_size/2
        self.cur_step = 0

        self.save_every = 5e5
        self.save_dir = save_dir


    def act(self, state):

        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).to(self.device)
            state = state.unsqueeze(0)
            action_values = self.online_net(state)
            action_idx = torch.argmax(action_values,axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min,self.exploration_rate)

        self.cur_step += 1

        return action_idx


    def memorize(self,state,next_state,action,reward,done):

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor([action])
        reward = torch.DoubleTensor([reward])
        done = torch.BoolTensor([done])

        self.memory.append((state, next_state, action, reward, done,))


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
        
        td_est = self.online_net(state)[np.arange(0, self.batch_size), action]

        with torch.no_grad():
            next_state_value = torch.max(self.target_net(next_state),1)[0]
            td_tgt = (reward + (1.0 - done.float())*self.gamma*next_state_value).float()

        loss = self.loss_fn(td_est, td_tgt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), td_est.mean().item()


    def sync_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())


    def observe(self,state,next_state,action,reward,done):
        
        self.memorize(state,next_state,action,reward,done)

        if self.cur_step % self.sync_every == 0:
            self.sync_target_net()

        if self.cur_step % self.save_every == 0:
            self.save()

        if self.memory.get_size() < self.mem_size_min:
            return None, None

        if self.cur_step % self.learn_every != 0:
            return None, None


        state, next_state, action, reward, done = self.recall()

        loss_val, td_est_mean = self.update(state, next_state, action, reward, done)


        return (loss_val, td_est_mean)

    
    def save(self):
        save_path = os.path.join(self.save_dir,"mario_net_{0}.chkpt".format(int(self.cur_step//self.save_every)))
        
        torch.save(
            dict(
                online_net=self.online_net.state_dict(),
                target_net=self.target_net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print("MarioNet saved to {0} at step {1}".format(save_path,self.cur_step))
    
