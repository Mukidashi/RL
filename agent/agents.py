from .DQN import DQN
from .doubleDQN import doubleDQN

import os


def get_agent(args,env):

    s_dim = env.observation_space.shape
    a_dim = env.action_space.n
    m_size = args.memory_size
    b_size = args.batch_size
    loss_type = args.loss_type

    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    agent = None
    if args.agent == "DQN":
        agent = DQN(state_dim=s_dim, action_dim=a_dim, save_dir=save_dir,
                    memory_size=m_size, batch_size=b_size, loss_type=loss_type)

    elif args.agent == "doubleDQN":
        
        agent = doubleDQN(state_dim=s_dim,action_dim=a_dim)
    
    return agent