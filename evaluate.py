from utils import *
import numpy as np
import random
import tensorflow as tf
import time
import os
import logging
import gym
from gym import envs, scoreboard
from gym.spaces import Discrete, Box
import prettytensor as pt
from space_conversion import SpaceConversionEnv
import tempfile
import sys
from trpo import TRPOAgent
from ur5_env import UR5Env

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        print('no model selected')
        exit(0)
    env = UR5Env("ur5_learn_position","ur5_control.launch")
    agent = TRPOAgent(env)
    agent.load_model(model)
    for i in xrange(10):
        #show 10 runs to check progress of the model
        agent.play_slow()
    env.close()
