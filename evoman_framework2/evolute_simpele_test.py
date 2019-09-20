import numpy as np

from matplotlib import pyplot as plt

from evolute import GeneticPopulation
from evolute.evaluation import SimpleFitness

TARGET = np.ones(10) * 0.5

pop = GeneticPopulation(loci=10,
                        fitness_wrapper=SimpleFitness(lambda ind: np.linalg.norm(ind - TARGET)))
"""
:param loci: number of elements in an individual's chromosome
:param fitness_wrapper: accepts an individual, returns fitnesses
:param limit: maximum number of individuals
:param operators: an instance of Operators
:param initializer: instance of a class defined in evolute.initialization
 and index of mutants
"""




history = pop.run(100, verbosity=1)
history = {k: np.array(v) for k, v in history.history.items()}

x = history["generation"]

plt.plot(x, history["mean_grade"], "r-", label="mean")
plt.plot(x, history["mean_grade"] + history["grade_std"], "b--", label="std")
plt.plot(x, history["mean_grade"] - history["grade_std"], "b--")
plt.plot(x, history["best_grade"], "g-", label="mean")

plt.title("Population convergence")
plt.legend()
plt.grid()
plt.show()
