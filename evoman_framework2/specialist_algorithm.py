
## Genetic algorithm
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
from controller import Controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os

experiment_name = 'first_run'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  enemymode="static",
                  level=2,
                  speed="fastest")
env.play()
