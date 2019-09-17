
## Genetic algorithm
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os

env = Environment(
                  enemies=[2],
                  playermode="ai",
                  enemymode="static",
                  level=2,
                  speed="fastest")
env.play()
