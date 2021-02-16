import numpy as np
import torch

from logger import EvalLogger

class Evaluate():
    
    def __init__(self, env, agent, args):
        self.env = env
        self.agent = agent

        self.save_dir = args.save_dir

        self.episode_num = args.eval_episode_num

        if args.checkpoint:
            self.agent.load(args.checkpoint)
        else:
            print("Note: CheckPoint Dir is not specified!")
    

    def process(self):
        self.env.reset()

        logger = EvalLogger(self.save_dir)

        for i in range(self.episode_num):
            state = self.env.reset()

            success = False
            while True:
                action = self.agent.act(state)

                next_state, reward, done, info = self.env.step(action)

                loss, qval = self.agent.eval(state, next_state, action, reward, done)

                screen = self.env.render()

                state = next_state

                logger.log_step(reward, loss, qval)

                if info['flag_get']:
                    success = True

                if done or info['flag_get']:
                    break
            
            logger.log_episode(i, success)

        logger.output_eval()

        self.env.close()