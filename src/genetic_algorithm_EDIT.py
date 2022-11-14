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

    def crossover(self, parent1: np.array, parent2: np.array) -> (np.array, np.array):
        """
        Performs a crossover operation on two parent chromosomes to product 2 child solutions
        :param parent1: n x m matrix representing chromosome of parent solution 1
        :param parent2: n x m matrix representing chromosome of parent solution 2
        :return: n x m matrix representing child solution 1,
                 n x m matrix representing child solution 2
        """
        # TODO: ADD CODE HERE
        pass

    def mutation(self, solution: np.ndarray, mutation_rate=None) -> np.array:
        """
        Randomly mutate genes in a solution with a low probability
        :param solution: n x m solution matrix to undergo mutation
        :param mutation_rate: the proportion of genes that will experience mutation at random
        :return: n x m mutated solution matrix
        """
        # TODO: ADD CODE HERE
        pass

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
        # TODO: ADD CODE HERE
        pass
