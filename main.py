
from train import Train
from evaluate import Evaluate

from env import get_env
from agent import get_agent

import argparse
import datetime


def set_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('proc_type',choices=['train','evaluate'],default='train')
    parser.add_argument('--env', choices=['cart','mario','bipedal_walker'], required=True)
    
    parser.add_argument('--agent',choices=['DQN','doubleDQN','duelingDQN','noisyDuelDQN','categoricalDoubleDQN','Rainbow','SAC'],required=True)
    parser.add_argument('--memory_size',type=int, default=100000)
    parser.add_argument('--memory_type', choices=['uniform','prioritized'], default='uniform')
    parser.add_argument('--batch_size',type=int, default=32)
    parser.add_argument('--loss_type',choices=['MSE','SmoothL1'],default="MSE")

    parser.add_argument('--episode_num',type=int, default=40000)
    parser.add_argument('--eval_episode_num', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default="./checkpoints/{0}".format(datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')))
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--eval_record', action='store_true')

    return parser

#Parser
parser = set_parser()
args = parser.parse_args()

#Setting
env = get_env(args.env)
agent = get_agent(args, env)

#Process
if args.proc_type == 'train':
    train = Train(env, agent, args)
    train.process()
else:
    evaluate = Evaluate(env, agent, args)
    evaluate.process()
