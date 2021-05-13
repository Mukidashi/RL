import gym


class AddCallbackFuncs(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)

    def is_end(self,done,info):
        return done
    
    def is_success(self, info):
        return info["success"]

    def get_action_dim(self):
        return self.action_space.shape

    def get_action_bound(self):
        return self.action_space.low, self.action_space.high
    
    def get_state_dim(self):
        return self.observation_space.shape

    def get_state_bound(self):
        return self.observation_space.low, self.observation_space.high


class BipedalWalker():
    
    def __init__(self,is_hard=False):
        self.is_hard = is_hard
    
    def get_env(self):
        env = gym.make('BipedalWalker-v3')
        env = AddCallbackFuncs(env)

        return env