from .DQN import DQN
from .doubleDQN import doubleDQN
from .duelingDQN import duelingDQN
from .categoricalDoubleDQN import categoricalDoubleDQN
from .Rainbow import Rainbow

import os


def get_agent(args,env):

    s_dim = env.observation_space.shape
    a_dim = env.action_space.n
    m_size = args.memory_size
    b_size = args.batch_size
    loss_type = args.loss_type
    mem_type = args.memory_type

    proc_type = args.proc_type

    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    agent = None
    if args.agent == "DQN":
        agent = DQN(state_dim=s_dim, action_dim=a_dim, save_dir=save_dir,
                    memory_size=m_size, batch_size=b_size, loss_type=loss_type, proc_type=proc_type)

    elif args.agent == "doubleDQN":
        
        agent = doubleDQN(state_dim=s_dim, action_dim=a_dim, save_dir=save_dir,
                           memory_size=m_size, memory_type=mem_type, batch_size=b_size, 
                           loss_type=loss_type, proc_type=proc_type)

    elif args.agent == "duelingDQN" or args.agent == "noisyDuelDQN":

        use_noisy = False
        if args.agent == "noisyDuelDQN":
            use_noisy = True
        
        agent = duelingDQN(state_dim=s_dim, action_dim=a_dim, save_dir=save_dir,
                           memory_size=m_size, memory_type=mem_type, batch_size=b_size, loss_type=loss_type,
                           use_noisy=use_noisy, proc_type=proc_type)

    elif args.agent == "categoricalDoubleDQN":

        agent = categoricalDoubleDQN(state_dim=s_dim, action_dim=a_dim, save_dir=save_dir,
                                     memory_size=m_size, memory_type=mem_type, batch_size=b_size, proc_type=proc_type)
    
    elif args.agent == "Rainbow":

        agent = Rainbow(state_dim=s_dim, action_dim=a_dim, save_dir=save_dir, 
                        memory_size=m_size, batch_size=b_size, proc_type=proc_type)

    return agent