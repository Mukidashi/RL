import numpy as np


class MinTree():
    
    def __init__(self, leaf_num):
        
        self.capacity = 1
        while self.capacity < leaf_num:
            self.capacity *= 2

        self.tree = np.zeros(self.capacity*2)


    def add(self, value, idx):
        idx = idx + self.capacity
        self.tree[idx] = value

        idx //= 2
        while idx >= 1:
            self.tree[idx] = min(self.tree[2*idx],self.tree[2*idx+1])
            idx //= 2


    def min(self,start=0,end=None):
        if end is None:
            end = self.capacity
        end -= 1
        return self._iterative_reducer(start, end, 1, 0, self.capacity - 1)

    
    def get_value(self, idx):
        return self.tree[idx + self.capacity]


    def _iterative_reducer(self, start, end, node, node_start, node_end):
        
        if start == node_start and end == node_end:
            return self.tree[node]
        
        mid = (node_start + node_end)//2
        if end <= mid:
            return self._iterative_reducer(start,end,2*node,node_start,mid)
        else:
            if mid + 1 <= start:
                return self._iterative_reducer(start, end, 2*node+1,mid+1,node_end)
            else:
                return min(self._iterative_reducer(start, mid, 2*node, node_start, mid), 
                            self._iterative_reducer(mid+1, end, 2*node+1, mid+1, node_end))
