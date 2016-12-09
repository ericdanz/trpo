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


if __name__ == "__main__":
    training_dir = tempfile.mkdtemp()
    logging.getLogger().setLevel(logging.DEBUG)

    if len(sys.argv) > 1:
        task = sys.argv[1]
    else:
        task = "Reacher-v1"
        # task = "CartPole-v1"

    env = envs.make(task)
    env.monitor.start(training_dir)

    agent = TRPOAgent(env)
    agent.learn()
    env.monitor.close()
    gym.upload(training_dir,
               algorithm_id='trpo_proximal_tf')
