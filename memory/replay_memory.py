from collections import deque
import random

class ReplayMemory(object):

    def __init__(self,capacity,batch_size=32):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
    
    def append(self,elem):
        self.memory.append(elem)

    def sample(self):
        batch = random.sample(self.memory,self.batch_size)
        return batch

    def get_size(self):
        return len(self.memory)