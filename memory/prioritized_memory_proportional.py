import numpy as np
import random
from memory.sum_tree import SumTree
from memory.min_tree import MinTree

class PrioritizedReplayMemoryProportional():

    def __init__(self, capacity, batch_size=32, priority_alpha=0.6):
        
        self.memory = []
        self.capacity = capacity
        self.next_id = 0
        self.batch_size = batch_size

        self.sum_tree = SumTree(capacity)
        self.min_tree = MinTree(capacity)

        self.priority_alpha = priority_alpha
        self.max_priority = 1.0
        self.epsilon = 1.0e-6

    
    def append(self, elem, priority):
        
        if self.next_id >= len(self.memory):
            self.memory.append(elem)
        else:
            self.memory[self.next_id] = elem
        
        priority = np.power(priority, self.priority_alpha)
        self.sum_tree.add(priority, self.next_id)
        self.min_tree.add(priority, self.next_id)

        self.next_id = (self.next_id + 1)%self.capacity


    def sample(self):

        samples = []
        sample_idx = []
        sample_prob = []

        sum_priority = self.sum_tree.sum()
        sample_width = sum_priority/float(self.batch_size)

        for i in range(self.batch_size):
            cumsum = random.random()*sample_width + i*sample_width
            idx = self.sum_tree.find_cumsum_idx(cumsum)
            
            samples.append(self.memory[idx])
            sample_idx.append(idx)
            sample_prob.append(self.sum_tree.get_value(idx)+self.epsilon)

        return samples, sample_idx, sample_prob
    

    def get_size(self):
        return len(self.memory)


    def get_max_priority(self):
        return self.max_priority


    def get_min_probability(self):
        return self.min_tree.min() + self.epsilon


    def update_priorities(self, idxs, vals):
        
        for i in range(len(idxs)):
            idx = idxs[i]
            val = vals[i]
            priority = np.power(val, self.priority_alpha)
            self.sum_tree.add(priority, idx)
            self.min_tree.add(priority, idx)

            self.max_priority = max(self.max_priority, val)