import heapq
import numpy as np
import random

from itertools import count

#Rank-Base
class PrioritizedReplayMemory():

    def __init__(self, capacity, batch_size=32, init_priority=1.0, priority_alpha=0.7):
        self.memory = []
        self.capacity = capacity
        self.batch_size = batch_size 
        self.init_priority = init_priority
        self.priority_alpha = priority_alpha
        
        self.section_idxs = [0]*(batch_size + 1)
        self.recalculate_thres = 0
        self.recalculate_step = self.capacity*0.1

        self.tiebreaker = count()


    def append(self, elem, priority):

        if len(self.memory) < self.capacity:
            heapq.heappush(self.memory,(-priority,next(self.tiebreaker), elem))
        else:
            self.memory.pop(-1)
            heapq.heappush(self.memory,(-priority,next(self.tiebreaker), elem))


    def sample(self):

        if len(self.memory) >= self.recalculate_thres:
            self.calculate_section()

        samples = []
        sample_idx = []
        sample_prob = []

        self.section_idxs[-1] = len(self.memory)
        for i in range(self.batch_size):
            if self.section_idxs[i] != self.section_idxs[i+1]:
                idx = random.randrange(self.section_idxs[i],self.section_idxs[i+1])
            else:
                idx = self.section_idxs[i]
            samples.append(self.memory[idx][2])
            sample_idx.append(idx)
            sample_prob.append(np.power(1.0/float(idx+1),self.priority_alpha))

        return samples, sample_idx, sample_prob
        

    def get_size(self):
        return len(self.memory)


    def get_max_priority(self):
        if len(self.memory) > 0:
            return -self.memory[0][0]
        else:
            return -self.init_priority


    def sort_memory(self):
        self.memory = [heapq.heappop(self.memory) for i in range(len(self.memory))]


    def calculate_section(self):
        
        if self.capacity >= self.recalculate_thres:
            self.recalculate_thres += self.recalculate_step

        rank = np.arange(len(self.memory)) + 1.0
        rank_recip = np.power(np.reciprocal(rank.astype(np.float32)),self.priority_alpha)
        rank_recip_cum = np.cumsum(rank_recip)

        section_width = (rank_recip_cum[-1]-1.0)/float(self.batch_size)
        section_bound = section_width + 1.0

        self.section_idxs[0] = 0
        sn = 1
        for i in range(len(self.memory)-1):
            if (rank_recip_cum[i] - section_bound)*(rank_recip_cum[i+1] - section_bound) <= 0.0:
                for j in range(sn,self.batch_size):
                    if(rank_recip_cum[i]-section_bound)*(rank_recip_cum[i+1]-section_bound) > 0.0:
                        break
                    self.section_idxs[sn] = i + 1
                    sn += 1
                    section_bound += section_width
                if sn == self.batch_size:
                    break


    def update_priorities(self, idxs, vals):

        for i in range(len(idxs)):
            idx = idxs[i]
            val = -vals[i]
            mlist = list(self.memory[idx])
            mlist[0] = val
            self.memory[idx] = tuple(mlist)

            pn = (idx-1)//2
            cn = 2*idx + 1
            if idx != 0 and val < self.memory[pn][0]:
                self.memory[idx], self.memory[pn] = self.memory[pn],self.memory[idx]
                if pn in idxs[i+1:]:
                    pid = idxs[i+1:].index(pn)
                    idxs[pid+i+1] = idx
                idxs[i+1:] = self.up_heap(pn,val,idxs[i+1:])
            elif cn < len(self.memory):
                if cn + 1 < len(self.memory) and self.memory[cn][0] > self.memory[cn+1][0]:
                    cn += 1
                if val > self.memory[cn][0]:
                    self.memory[idx], self.memory[cn] = self.memory[cn], self.memory[idx]
                    if cn in idxs[i+1:]:
                        cid = idxs[i+1:].index(cn)
                        idxs[cid+i+1] = idx
                    idxs[i+1:] = self.down_heap(cn,val,idxs[i+1:])


    def up_heap(self, idx, val, trace_idxs):

        if idx == 0:
            return trace_idxs
        
        pn = (idx - 1)//2
        if val < self.memory[pn][0]:
            self.memory[idx], self.memory[pn] = self.memory[pn], self.memory[idx]
            if pn in trace_idxs:
                pid = trace_idxs.index(pn)
                trace_idxs[pid] = idx
            trace_idxs = self.up_heap(pn, val, trace_idxs)
        return trace_idxs


    def down_heap(self, idx, val, trace_idxs):

        if 2*idx  + 1 >= len(self.memory):
            return trace_idxs

        cn = 2*idx + 1
        if cn + 1 < len(self.memory) and self.memory[cn][0] > self.memory[cn+1][0]:
            cn += 1
        if val > self.memory[cn][0]:
            self.memory[idx], self.memory[cn] = self.memory[cn], self.memory[idx]
            if cn in trace_idxs:
                cid = trace_idxs.index(cn)
                trace_idxs[cid] = idx
            trace_idxs = self.down_heap(cn, val,trace_idxs)
        return trace_idxs

        