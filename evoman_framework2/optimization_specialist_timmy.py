###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os


experiment_name = 'tim_testerino'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  sound="off")

# default environment fitness is assumed for experiment

#env.state_to_log() # checks environment state

# runs simulation for player x
def simulation(env,x):
    fitness, hp_player, hp_enemy, timePlayed = env.play(pcont=x)
    return fitness


from evolute import GeneticPopulation
from evolute.evaluation import SimpleFitness

n_hidden = 10 #dit is het aantal hidden neurons in onze demo_controller
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 # #elementen in 1 chromosoom
#heb dit ff hier neergezet om die loci=265 te verklaren


pop = GeneticPopulation(loci=n_vars, limit=9,
                        fitness_wrapper=SimpleFitness(lambda x: simulation(env,x)))

history = pop.run(20, verbosity=1)

history = {k: np.array(v) for k, v in history.history.items()}

x = history["generation"]

from matplotlib import pyplot as plt

plt.plot(x, history["mean_grade"], "r-", label="mean")
plt.plot(x, history["mean_grade"] + history["grade_std"], "b--", label="std")
plt.plot(x, history["mean_grade"] - history["grade_std"], "b--")
plt.plot(x, history["best_grade"], "g-", label="best")

plt.title("Population convergence")
plt.legend()
plt.grid()
plt.show()
