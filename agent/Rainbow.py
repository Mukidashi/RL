import torch
import numpy as np
import os


from memory import ReplayMemory, PrioritizedReplayMemoryProportional
from network import MarioRainbowNet

def CE_loss(x,y):
    loss = -x*torch.log(y+1.0e-8)
    return torch.sum(loss, 1)

class Rainbow:

    def __init__(self, state_dim, action_dim, save_dir, memory_size=100000,
                 batch_size=32, proc_type="train"):
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = batch_size
        self.memory = PrioritizedReplayMemoryProportional(memory_size, batch_size=batch_size, priority_alpha=0.5)

        self.sample_idx = None
        self.sample_prob = None
        self.beta_zero = 0.4
        self.beta = self.beta_zero
        self.anneal_IS_every = 100000
        self.anneal_IS_step = (1.0 - self.beta_zero)/100.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.atom_num = 51
        self.online_net = MarioRainbowNet(state_dim, action_dim, self.atom_num).to(self.device)
        self.target_net = MarioRainbowNet(state_dim, action_dim, self.atom_num).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.Vmin = 0.0
        self.Vmax = 100.0
        self.atoms = np.zeros(self.atom_num)
        self.deltaz = (self.Vmax - self.Vmin)/float(self.atom_num-1)
        for i in range(self.atom_num):
            self.atoms[i] = self.Vmin + float(i)*self.deltaz

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=0.00025/4.0)

        self.loss_fn = CE_loss

        self.proc_type = proc_type

        self.exploration_rate = -1.0

        self.learn_every = 3
        self.sync_every = 1e4
        self.mem_size_min = memory_size/2
        self.cur_step = 0
        
        self.gamma = 0.9
        self.multi_step_num = 3
        self.exp_stack = [None]*self.multi_step_num
        self.reward_stack = np.zeros(self.multi_step_num)
        self.multi_step_idx = 0
        self.n_gamma = np.power(self.gamma, self.multi_step_num)

        self.save_every = 5e5
        self.save_dir = save_dir


    def act(self, state):

        self.online_net.sample_noise()

        state = torch.FloatTensor(state).to(self.device)
        state = state.unsqueeze(0)
        action_values = self._convert_categorical_output_to_Qvalue(self.online_net(state))
        action_idx = torch.argmax(action_values, axis=1).item()

        self.cur_step += 1

        return action_idx


    def memorize(self, state, next_state, action, reward, done):

        self.exp_stack[self.multi_step_idx] = (state, action)
        self.reward_stack[self.multi_step_idx] = reward

        self.multi_step_idx = (self.multi_step_idx+1)%self.multi_step_num
        
        if self.exp_stack[self.multi_step_idx] is None:
            return
        
        n_reward = 0.0
        for i in range(self.multi_step_num):
            idx = (self.multi_step_idx+i)%self.multi_step_num
            n_reward += np.power(self.gamma,i)*self.reward_stack[idx]

        state, action = self.exp_stack[self.multi_step_idx]
        
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor([action])
        reward = torch.DoubleTensor([n_reward])
        done = torch.BoolTensor([done])

        max_priority = self.memory.get_max_priority()
        self.memory.append((state, next_state, action, reward, done,), max_priority)


    def recall(self):
        
        batch, self.sample_idx, self.sample_prob = self.memory.sample()
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))

        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


    def update(self, state, next_state, action, reward, done):
        
        with torch.no_grad():

            self.online_net.sample_noise()
            self.target_net.sample_noise()

            zsupp = torch.tensor([self.Vmin + float(i)*self.deltaz for i in range(self.atom_num)]).to(self.device)
            Tz = reward.view(-1,1).repeat(1,self.atom_num) + (1.0 - done.float().view(-1,1))*self.n_gamma*zsupp.view(1,-1).repeat(self.batch_size,1)
            # Tz = reward.view(-1, 1).repeat(1, self.atom_num) + self.n_gamma*zsupp.view(1,-1).repeat(self.batch_size, 1)
            Tz = torch.clip(Tz, self.Vmin, self.Vmax)
            Tz_n = (Tz-self.Vmin)/float(self.deltaz)
            Tz_lid = torch.floor(Tz_n).type(torch.long)
            Tz_uid = torch.ceil(Tz_n).type(torch.long)

            max_action_id = torch.argmax(self._convert_categorical_output_to_Qvalue(self.online_net(next_state)), axis=1)
            tgt_prob = self.target_net(next_state)[np.arange(0,self.batch_size),max_action_id]
            
            lvals = tgt_prob*(Tz_uid-Tz_n)
            uvals = tgt_prob*(Tz_n-Tz_lid)
            tgt_dist = torch.zeros(self.batch_size,self.atom_num).to(self.device).scatter_(1, Tz_lid, lvals.type(torch.float), reduce="add") \
                       + torch.zeros(self.batch_size, self.atom_num).to(self.device).scatter_(1, Tz_uid, uvals.type(torch.float), reduce="add")
        
        self.online_net.sample_noise()

        est_dist = self.online_net(state)[np.arange(0,self.batch_size),action]

        wei = np.power(np.reciprocal(np.array(self.sample_prob)),self.beta)
        wei = wei/np.amax(wei)
        wei = torch.FloatTensor(wei).to(self.device)

        ce_loss = self.loss_fn(tgt_dist, est_dist)

        new_priorities = ce_loss.to('cpu').detach().numpy().copy()
        self.memory.update_priorities(self.sample_idx,new_priorities)

        loss = torch.mean(wei*ce_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        q_est = self._convert_categorical_output_to_Qvalue(est_dist)

        return loss.item(), q_est.mean().item()


    @torch.no_grad()
    def eval(self, state, next_state, action, reward, done, is_end):
        state = torch.FloatTensor(state).to(self.device)
        state = state.unsqueeze(0)
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_state = next_state.unsqueeze(0)
        reward = torch.DoubleTensor([reward]).to(self.device)

        self.online_net.sample_noise()
        self.target_net.sample_noise()

        zsupp = torch.tensor([self.Vmin + float(i)*self.deltaz for i in range(self.atom_num)]).to(self.device)
        Tz = reward.view(-1,1).repeat(1,self.atom_num) + (1.0-float(done))*self.gamma*zsupp.view(1,-1)
        # Tz = reward.view(-1, 1).repeat(1, self.atom_num) + self.gamma*zsupp.view(1, -1)
        Tz = torch.clip(Tz, self.Vmin, self.Vmax)
        Tz_n = (Tz-self.Vmin)/float(self.deltaz)
        Tz_lid = torch.floor(Tz_n).type(torch.long)
        Tz_uid = torch.ceil(Tz_n).type(torch.long)

        max_action_id = torch.argmax(self._convert_categorical_output_to_Qvalue(self.online_net(next_state)),axis=1)
        tgt_prob = self.target_net(next_state)[:,max_action_id].view(1,-1)

        lvals = tgt_prob*(Tz_uid - Tz_n)
        uvals = tgt_prob*(Tz_n - Tz_lid)
        tgt_dist = torch.zeros(1, self.atom_num).to(self.device).scatter_(1, Tz_lid, lvals.type(torch.float), reduce="add") \
                    + torch.zeros(1, self.atom_num).to(self.device).scatter_(1, Tz_uid, uvals.type(torch.float), reduce="add")


        self.online_net.sample_noise()
        
        est_dist = self.online_net(state)[:, action]
        q_est = self._convert_categorical_output_to_Qvalue(est_dist)

        loss = torch.mean(self.loss_fn(tgt_dist, est_dist))

        return loss.item(), q_est.mean().item()


    def sync_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    
    def observe(self, state, next_state, action, reward, done, is_end):

        self.memorize(state, next_state, action, reward, done)

        if is_end:
            self.exp_stack = [None]*self.multi_step_num
            self.reward_stack = np.zeros(self.multi_step_num)
            self.multi_step_idx = 0

        if self.cur_step % self.sync_every == 0:
            self.sync_target_net()

        if self.cur_step % self.save_every == 0:
            self.save()

        if self.memory.get_size() < self.mem_size_min:
            return None, None

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
                is_beta=self.beta
            ),
            save_path
        )
        print("MarioNet saved to {0} at step {1}".format(save_path,self.cur_step))


    def load(self, load_path):
        if not os.path.exists(load_path):
            raise ValueError(f"{load_path} not Exist")
        
        ckp = torch.load(load_path,map_location=self.device)
        online_net = ckp.get('online_net')
        is_beta = ckp.get('is_beta')

        self.online_net.load_state_dict(online_net)
        self.target_net.load_state_dict(online_net)
        self.beta = is_beta

        print(f"Loading model at {load_path}")


    def anneal_IS_beta(self):
        if self.beta <= 1.0:
            self.beta += self.anneal_IS_step
        print("Aneal IS beta:{0}".format(self.beta))


    def _convert_categorical_output_to_Qvalue(self, output):
        atoms = torch.tensor(self.atoms).type(torch.float).to(self.device)
        qvals = torch.matmul(output,atoms)
        return qvals