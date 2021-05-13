import numpy as np
import torch

from logger import Logger, SACLogger

class Train():
    
    def __init__(self, env, agent, args):
        self.env = env
        self.agent = agent
        self.agent_name = args.agent

        self.save_dir = args.save_dir

        self.episode_num = args.episode_num

        if args.checkpoint:
            self.agent.load(args.checkpoint)

    def process(self):
        self.env.reset()

        if self.agent_name != "SAC":
            logger = Logger(self.save_dir)
        else:
            logger = SACLogger(self.save_dir)

        for i in range(self.episode_num):
            state = self.env.reset()

            while True:
                action = self.agent.act(state)

                next_state, reward, done, info = self.env.step(action)

                is_end = self.env.is_end(done,info)

                outs = self.agent.observe(state, next_state, action, reward, done, is_end)

                screen = self.env.render()

                state = next_state

                logger.log_step(reward, outs)

                if is_end:
                    break
            
            logger.log_episode()

            if i%20 == 0:
                logger.record(i, self.agent.exploration_rate, self.agent.cur_step)

        self.env.close()
        

