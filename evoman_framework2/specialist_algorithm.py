
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


from evolute import GeneticPopulation
from evolute.evaluation import SimpleFitness

experiment_name = 'first_run'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  enemymode="static",
                  level=2,
                  speed="fastest")

x = 10
def simulation(env,x):
  fitness, hp_player, hp_enemy, time = env.play(pcont=x)
  return fitness

pop = GeneticPopulation(loci=265, limit=5,
                        fitness_wrapper=SimpleFitness(lambda x: simulation(env,x)))


history = pop.run(10, verbosity=1)
