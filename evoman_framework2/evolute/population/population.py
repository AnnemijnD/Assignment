import numpy as np

from ..initialization import DefaultInitializer
from ..operators import Operators
from ..evaluation import SimpleFitness
from ..utility.history import History


class Population:

    def __init__(self, loci: int,
                 fitness_wrapper,
                 limit=10,
                 operators=None,
                 initializer=None):
        """
        :param loci: number of elements in an individual's chromosome
        :param fitness_wrapper: accepts an individual, returns fitnesses
        :param limit: maximum number of individuals
        :param operators: an instance of Operators
        :param initializer: instance of a class defined in evolute.initialization
         and index of mutants
        """
        self.loci = loci
        self.limit = limit
        self.fitness = fitness_wrapper
        self.fitnesses = None
        self.operators = Operators() if operators is None else operators

        self.initializer = DefaultInitializer() if initializer is None else initializer
        self.individuals = self.initializer.initialize(self.limit, self.loci)

        self.age = 0
        self.champion = 0

    @classmethod
    def simple_fitness(cls, fitness_callback,
                       loci, limit=3,
                       initializer=None,
                       operators=None,
                       fitness_constants=None):
        fitness_wrapper = SimpleFitness(fitness_callback, {} if fitness_constants is None else fitness_constants)
        return cls(loci=loci, fitness_wrapper=fitness_wrapper, limit=limit,
                   initializer=initializer, operators=operators)

    def get_individual(self, index):
        return self.individuals[index]

    def set_individual(self, index, individual):
        self.individuals[index] = individual

    def get_best(self):
        return self.get_individual(np.argmin(self.fitnesses))

    def get_champion(self):
        return self.get_individual(self.champion)

    def get_fitness_weighted_average_individual(self):
        weights = (self.fitnesses - self.fitnesses.mean()) / self.fitnesses.std()
        return weights @ self.individuals

    def run(self, epochs: int,
            survival_rate: float=0.5,
            mutation_rate: float=0.1,
            force_update_at_every: int=999999,
            verbosity: int=1,
            history=None):
        """
        :param epochs: number of epochs to run for
        :param survival_rate: 0-1, how many individuals survive the selection
        :param mutation_rate: 0-1, rate of mutation at each epoch
        :param force_update_at_every: complete reupdate at specified intervals
        :param verbosity: 1 is verbose, > 1 also prints out v - 1 individuals
        :param history: History object in which run stats should be recorded
        :return: history object containing run statistics
        """

        history = (History(["generation", "best_grade", "mean_grade", "grade_std"])
                   if history is None else history)
        self.operators.selection.set_survival_rate(survival_rate)
        self.operators.mutation.set_rate(mutation_rate)
        for epoch in range(1, epochs+1):
            if verbosity:
                print("-" * 50)
                print("Epoch {}/{}".format(epochs, epoch))
            self.epoch(force_update=force_update_at_every and epoch % force_update_at_every == 0,
                       verbosity=verbosity)
            history.record({"generation": self.age,
                            "best_grade": self.fitnesses.max(),
                            "mean_grade": self.fitnesses.mean(),
                            "grade_std": self.fitnesses.std()})
        if verbosity:
            print()
        return history

    def epoch(self, force_update=False, verbosity=1, **fitness_kw):
        if not self.age:
            self._initialize(verbosity, **fitness_kw)

        self.operators.selection(self.individuals, self.fitnesses, inplace=True)
        self.individuals = self.operators.mutation(self.individuals, inplace=False)
        self.update(force_update, verbose=verbosity, **fitness_kw)
        self.age += 1

    def _initialize(self, verbosity, **fitness_kw):
        if verbosity:
            print("EVOLUTION: initial update...")
        self.fitnesses = np.empty(self.limit)
        self.update(forced=True, verbose=verbosity, **fitness_kw)
        if verbosity:
            print("EVOLUTION: initial mean grade :", self.fitnesses.mean())
            print("EVOLUTION: initial std of mean:", self.fitnesses.std())
            print("EVOLUTION: initial best grade :", self.fitnesses.max())

    def update(self, forced=False, verbose=0, **fitness_kw):
        inds = self._invalidated_individual_indices(force_update=forced)
        for ind in inds.flat:
            if verbose:
                print("\rUpdating {}/{}".format(ind+1, self.limit), end="")
            self.update_individual(ind, **fitness_kw)
        if verbose:
            print("\rUpdating {}/{}".format(self.limit, self.limit), end="")
            print(" Best grade:", self.fitnesses.max())
        chump = self.fitnesses.argmax()
        if self.fitnesses[chump] > self.fitnesses[self.champion]:
            self.champion = chump

    def update_individual(self, index, **fitness_kw):
        raise NotImplementedError 

    def _invalidated_individual_indices(self, force_update):
        returnDit = np.arange(self.limit) if force_update else self.operators.invalid_individual_indices()
        print("Dit zijn de indices van de levende individuen {}".format(returnDit))
        # deze verwijst naar np.where(self.selection.mask | self.mutation.mask)[0]
        # dat is goed, maar ik snap niet waar de populatie weer opgevuld wordt...
        return returnDit

    def save(self, path):
        import pickle
        import gzip
        with gzip.open(path, "wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path):
        import pickle
        import gzip
        return pickle.load(gzip.open(path, "rb"))
