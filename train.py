import numpy as np
import torch

from logger import Logger

class Train():
    
    def __init__(self, env, agent, args):
        self.env = env
        self.agent = agent

        self.save_dir = args.save_dir

        self.episode_num = args.episode_num

    def process(self):
        self.env.reset()

        logger = Logger(self.save_dir)

        for i in range(self.episode_num):
            state = self.env.reset()

            while True:
                action = self.agent.act(state)

                next_state, reward, done, info = self.env.step(action)

                loss, qval = self.agent.observe(state, next_state, action, reward, done)

                screen = self.env.render()

                state = next_state

                logger.log_step(reward, loss, qval)

                if done or info['flag_get']:
                    break
            
            logger.log_episode()

            if i%20 == 0:
                logger.record(episode = i,
                              epsilon = self.agent.exploration_rate,
                              step = self.agent.cur_step)

        self.env.close()
        

