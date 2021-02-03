import gym

from .cart_env import *
from .mario_env import *


def get_env(env_name):
    env = None
    if env_name == "cart":
        env = CartEnv().get_env()
    elif env_name == "mario":
        env = MarioEnv().get_env()
    
    return env

