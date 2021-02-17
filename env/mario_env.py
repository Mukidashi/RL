#This code is based on the official tutorial of pytorch

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

import numpy as np

from env.env_util import SkipFrame, ResizeObservation, AddEndChecker



class MarioEnv():
    def __init__(self, skip_frame=4, img_shape=(84,84), frame_stack=4):
        self.skipN = skip_frame
        self.stackN = frame_stack
        self.img_shape = img_shape

    def get_env(self):
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

        env = JoypadSpace(env,[['right'],['right','A']])

        env = SkipFrame(env,skip=self.skipN)
        env = GrayScaleObservation(env, keep_dim=False)
        env = ResizeObservation(env,shape=self.img_shape)
        env = TransformObservation(env, f=lambda x:x/255.)
        env = FrameStack(env,num_stack=self.stackN)
        env = AddEndChecker(env)
        
        env.observation_space.low = np.zeros_like(env.observation_space.low)
        env.observation_space.high = np.zeros_like(env.observation_space.high) + 1.0
        env.observation_space.dtype = np.float32

        return env