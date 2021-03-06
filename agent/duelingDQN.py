import torch
import numpy as np
import os


from memory import ReplayMemory, PrioritizedReplayMemoryProportional
from network import MarioDuelNet, MarioNoisyDuelNet

def weighted_MSE_Loss(x,y,w):
    loss = w*((x-y)**2)
    return torch.mean(loss)

class duelingDQN:

    def __init__(self, state_dim, action_dim, save_dir, memory_size=100000,memory_type='uniform',
                 batch_size=32, loss_type="MSE", use_noisy=False, proc_type="train"):
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = batch_size
        self.memory_type = memory_type
        if memory_type == 'prioritized':
            self.memory = PrioritizedReplayMemoryProportional(memory_size, batch_size=self.batch_size)
        else:
            self.memory = ReplayMemory(memory_size, batch_size=self.batch_size)

        if memory_type == 'prioritized':
            # self.sort_memory_every = memory_size
            self.sample_idx = None
            self.sample_prob = None
            self.beta_zero = 0.4
            self.beta = self.beta_zero
            self.anneal_IS_every = 100000
            self.anneal_IS_step = (1.0-self.beta_zero)/100.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        

        self.use_noisy = use_noisy
        if use_noisy:
            self.online_net = MarioNoisyDuelNet(state_dim, action_dim).to(self.device)
            self.target_net = MarioNoisyDuelNet(state_dim, action_dim).to(self.device)
        else:
            self.online_net = MarioDuelNet(state_dim,action_dim).to(self.device)
            self.target_net = MarioDuelNet(state_dim,action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=0.00025/4.0)

        if loss_type == "SmoothL1":
            self.loss_fn = torch.nn.SmoothL1Loss()
        else:
            self.loss_fn = torch.nn.MSELoss()

        if memory_type == 'prioritized' and proc_type == "train":
            #for now, only one function is implemented for prioritized replay
            self.loss_fn = weighted_MSE_Loss

        
        self.exploration_rate = 1
        # self.exploration_rate_decay = 0.99999975
        self.exploration_rate_step = 0.9/4500000.0
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.proc_type = proc_type
        if self.proc_type == "evaluate":
            self.exploration_rate = 0.05
            self.exploration_min = 0.05

        if self.use_noisy:
            self.exploration_rate = -1

        self.learn_every = 3
        self.sync_every = 1e4
        self.mem_size_min = memory_size/2
        self.cur_step = 0

        self.save_every = 5e5
        self.save_dir = save_dir


    def act(self, state):

        if self.use_noisy:
            self.online_net.sample_noise()

        #epsilon-greedy (if model is noisy, never select random policy)
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).to(self.device)
            state = state.unsqueeze(0)
            action_values = self._convert_duel_output_to_Qvalue(self.online_net(state))
            action_idx = torch.argmax(action_values,axis=1).item()

        if not self.use_noisy and self.proc_type == "train":
            # self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate -= self.exploration_rate_step
            self.exploration_rate = max(self.exploration_rate_min,self.exploration_rate)

        self.cur_step += 1

        return action_idx


    def memorize(self,state,next_state,action,reward,done):

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor([action])
        reward = torch.DoubleTensor([reward])
        done = torch.BoolTensor([done])

        if self.memory_type == "prioritized":
            max_priority = self.memory.get_max_priority()
            self.memory.append((state, next_state, action, reward, done,),max_priority)
        else:
            self.memory.append((state, next_state, action, reward, done,))


    def recall(self):
        if self.memory_type == "prioritized":
            batch, self.sample_idx, self.sample_prob = self.memory.sample()
        else:
            batch = self.memory.sample()
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))

        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)
        
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

        
    def update(self, state, next_state, action, reward, done):

        with torch.no_grad():
            if self.use_noisy:
                self.online_net.sample_noise()
                self.target_net.sample_noise()

            q_next_est = self._convert_duel_output_to_Qvalue(self.online_net(next_state))
            max_action_id = torch.argmax(q_next_est, axis=1)
            q_next_tgt = self._convert_duel_output_to_Qvalue(self.target_net(next_state))
            next_state_value = q_next_tgt[np.arange(0, self.batch_size),max_action_id]
            td_tgt = (reward + (1.0 - done.float())*self.gamma*next_state_value).float()


        if self.use_noisy:
            self.online_net.sample_noise()

        q_est = self._convert_duel_output_to_Qvalue(self.online_net(state))
        td_est = q_est[np.arange(0, self.batch_size), action]


        if self.memory_type == 'prioritized':
            td_error = td_est - td_tgt
            td_abs_error = np.abs(td_error.view(self.batch_size).to('cpu').detach().numpy().copy())
            self.memory.update_priorities(self.sample_idx,td_abs_error)

            wei = np.power(np.reciprocal(np.array(self.sample_prob)),self.beta)
            wei = wei/np.amax(wei)
            wei = torch.FloatTensor(wei).to(self.device)

            loss = self.loss_fn(td_est, td_tgt, wei)

        else:
            loss = self.loss_fn(td_est, td_tgt)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), td_est.mean().item()


    def eval(self, state, next_state, action, reward, done, is_end):
        
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            state = state.unsqueeze(0)
            next_state = torch.FloatTensor(next_state).to(self.device)
            next_state = next_state.unsqueeze(0)

            if self.use_noisy:
                self.online_net.sample_noise()
                self.target_net.sample_noise()

            q_next_est = self._convert_duel_output_to_Qvalue(self.online_net(next_state))
            max_action_id = torch.argmax(q_next_est, axis=1)
            q_next_tgt = self._convert_duel_output_to_Qvalue(self.target_net(next_state))
            next_state_value = q_next_tgt[:,max_action_id]
            td_tgt = (reward + (1.0 - float(done))*self.gamma*next_state_value).float()

            if self.use_noisy:
                self.online_net.sample_noise()
            q_est = self._convert_duel_output_to_Qvalue(self.online_net(state))
            td_est = q_est[:, action]

            loss = self.loss_fn(td_est, td_tgt)

        return loss.item(), td_est.mean().item()


    def sync_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())


    def observe(self, state, next_state, action, reward, done, is_end):
        
        self.memorize(state,next_state,action,reward,done)

        if self.cur_step % self.sync_every == 0:
            self.sync_target_net()

        if self.cur_step % self.save_every == 0:
            self.save()

        if self.memory.get_size() < self.mem_size_min:
            return None, None

        # if self.memory_type == "prioritized" and self.cur_step % self.sort_memory_every == 0:
        #     self.memory.sort_memory()

        if self.cur_step % self.anneal_IS_every == 0:
            self.anneal_IS_beta()

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
                exploration_rate=self.exploration_rate,
                is_beta=self.beta
            ),
            save_path
        )
        print("MarioNet saved to {0} at step {1}".format(save_path,self.cur_step))


    def load(self, load_path):
        if not os.path.exists(load_path):
            raise ValueError(f"{load_path} not Exist")
        
        ckp = torch.load(load_path,map_location=self.device)
        exploration_rate = ckp.get('exploration_rate')
        online_net = ckp.get('online_net')
        is_beta = ckp.get('is_beta')

        self.online_net.load_state_dict(online_net)
        self.target_net.load_state_dict(online_net)
        self.beta = is_beta

        if self.proc_type == "train":
            self.exploartion_rate = exploration_rate

        print(f"Loading model at {load_path} with exploration rate {self.exploration_rate}")


    def anneal_IS_beta(self):
        if self.beta <= 1.0:
            self.beta += self.anneal_IS_step
        print("Aneal IS beta:{0}".format(self.beta))


    def _convert_duel_output_to_Qvalue(self,output):
        value, advantage = output
        advantage_mean = torch.mean(advantage,1,True)
        qvalue = value.repeat((1,self.action_dim)) + advantage - advantage_mean.repeat((1,self.action_dim))
        return qvalue
        
    
