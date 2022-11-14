# Genetic Algorithm class
from problem import *
from typing import Callable
import numpy.random as random
from base import BaseGA


class GA(BaseGA):
    """
    Genetic Algorithm class
    Stateless - does not store GA solution populations internally
    Initialised with objective & constraint functions, penalty weights and optimisation parameters
    Adds facility for including design knowledge in initial population
    """

    def __init__(self,
                 minimisation_objective_func: Callable[[np.ndarray], float],
                 named_constraint_funcs: dict,
                 constraint_penalty_weights: np.array,
                 pop_size: int,
                 elitism_size: int,
                 mutation_rate: float,
                 decision_var_shape: (int, int),
                 random_init_func: Callable[[int, int, int], np.ndarray],
                 include_design_knowledge: bool = False,
                 ):
        """
        Initialise the Genetic Algorithm class (base)
        @param minimisation_objective_func: function to be minimised that takes np.array input
        @param named_constraint_funcs: dictionary of named constraint functions that are added as penalty terms
        @param constraint_penalty_weights: vector of objective function penalty weights for constraint terms
        @param pop_size: number of individual solutions in GA population
        @param elitism_size: number of top solutions carried forward to next generation
        @param mutation_rate: the proportion of genes undergoing mutation
        @param decision_var_shape: the shape of the decision variable matrix
        @param random_init_func: function for randomly initialising the GA population
        @param include_design_knowledge: if True include solutions informed by design knowledge in init population
        """
        super().__init__(minimisation_objective_func, named_constraint_funcs, constraint_penalty_weights,
                         pop_size, elitism_size, mutation_rate, decision_var_shape, random_init_func)
        self.include_design_knowledge = include_design_knowledge
        self.const_row1 = np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1])
        self.const_row2 = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1])
        self.const_row3 = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1])
        self.const_row4 = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
        self.const_row5 = np.array([1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1])

    def get_constructed_row(self):
        randi = random.randint(1, 5)
        if randi == 1:
            return self.const_row1
        elif randi == 2:
            return self.const_row2
        elif randi == 3:
            return self.const_row3
        elif randi == 4:
            return self.const_row4
        elif randi == 5:
            return self.const_row5

    def random_initialisation(self):
        """
        Randomly initialise the population of chromosomes
        :return: n x m x pop_size matrix of 0's and 1's
        """
        rand_pop = self.random_init_func(self.n, self.m, self.pop_size)
        if self.include_design_knowledge:
            rand_pop[:, :, 0] = np.concatenate([[self.get_constructed_row()] for _ in range(self.n)], axis=0)
            rand_pop[:, :, 1] = np.concatenate([[self.get_constructed_row()] for _ in range(self.n)], axis=0)
            rand_pop[:, :, 2] = np.ones((self.n, self.m))
            rand_pop[:, :, 3] = np.zeros((self.n, self.m))
        return rand_pop

    def crossover(self, parent1: np.array, parent2: np.array) -> (np.array, np.array):
        """
        Performs a crossover operation on two parent chromosomes to product 2 child solutions
        :param parent1: n x m matrix representing chromosome of parent solution 1
        :param parent2: n x m matrix representing chromosome of parent solution 2
        :return: n x m matrix representing child solution 1,
                 n x m matrix representing child solution 2
        """
        idx = random.randint(1, parent1.shape[0] - 1)
        child1 = np.concatenate((parent1[:idx], parent2[idx:]))
        child2 = np.concatenate((parent2[:idx], parent1[idx:]))
        return child1, child2

    def mutation(self, solution: np.ndarray, mutation_rate=None) -> np.array:
        """
        Randomly mutate genes in a solution with a low probability
        :param solution: n x m solution matrix to undergo mutation
        :param mutation_rate: the proportion of genes that will experience mutation at random
        :return: n x m mutated solution matrix
        """
        total_entries = solution.shape[0] * solution.shape[1]
        total_to_mutate = int(total_entries * self.mutation_rate)
        for mut in range(total_to_mutate):
            idx_n = random.choice(solution.shape[0], size=None, replace=True, p=None)
            idx_m = random.choice(solution.shape[1], size=None, replace=True, p=None)
            solution[idx_n, idx_m] = 1 - solution[idx_n, idx_m]
        return solution

    # uses an elitist selection mechanism to return best of two parent solutions
    def selection(self,
                  population: np.ndarray,
                  population_fitness: np.array) -> np.array:
        """
        Fitness biased selection of a single parent solution
        :param population: the n x m x pop_size matrix of solutions
        :param population_fitness: the pop_size length array of solution fitnesses
        :return: n x m array parent solution
        """
        # create a probability of selection by dividing total fitness
        selection_probability = population_fitness / sum(population_fitness)
        # create cumulative probabilities per solution
        cumulative_probability = [sum(selection_probability[:i + 1]) for i in range(len(selection_probability))]
        # select index with the calculated probabilities
        rand = random.random()
        selected_index = len([c for c in cumulative_probability if c < rand])
        selected = population[:, :, selected_index]
        return selected
