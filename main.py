
from train import Train
from evaluate import Evaluate
from logger import Logger

from env import get_env
from agent import get_agent

import argparse
import datetime


def set_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('proc_type',choices=['train','evaluate'],default='train')
    parser.add_argument('--env', choices=['cart','mario'], required=True)
    
    parser.add_argument('--agent',choices=['DQN','doubleDQN'],required=True)
    parser.add_argument('--memory_size',type=int, default=100000)
    parser.add_argument('--memory_type', choices=['uniform','prioritized'], default='uniform')
    parser.add_argument('--batch_size',type=int, default=32)
    parser.add_argument('--loss_type',choices=['MSE','SmoothL1'],default="MSE")

    parser.add_argument('--episode_num',type=int,default=40000)
    parser.add_argument('--save_dir',type=str, default="./checkpoints/{0}".format(datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')))

    return parser


parser = set_parser()
args = parser.parse_args()

env = get_env(args.env)
agent = get_agent(args, env)

if args.proc_type == 'train':
    Train(args).process()
else:
    Evaluate(args).process()



#test
env.reset()

import numpy as np 
import torch

logger = Logger(args.save_dir)

action_num = env.action_space.n
episode_num = args.episode_num
for i in range(episode_num):
    
    state = env.reset()

    # iterN = 0
    # reward_sum = 0.0
    # loss_sum = 0.0
    # loss_cnt = 0
    while True:

        # action = np.random.randint(action_num)
        action = agent.act(state)
        
        next_state, reward, done, info = env.step(action)

        loss, q = agent.observe(state, next_state, action, reward, done)

        screen = env.render()

        state = next_state
         
        # if loss:
        #     loss_sum += loss
        # reward_sum += reward
        # iterN += 1

        logger.log_step(reward,loss,q)

        if done or info['flag_get']:
            break

    if args.memory_type == 'prioritized':
        agent.anneal_IS_beta(float(i)/float(episode_num))

    logger.log_episode()
    
    # print("Episode{0}-->iter:{1} loss:{2} reward:{3}".format(i,iterN,loss_sum/float(iterN),reward_sum))

    if i%20 == 0:
        logger.record(episode=i,
                      epsilon=agent.exploration_rate,
                      step=agent.cur_step)

env.close()
