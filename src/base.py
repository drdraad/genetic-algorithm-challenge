# Base GA class
from abc import ABC, abstractmethod
from typing import Callable
import numpy as np


class BaseGA(ABC):
    """
    Base class for a stateless genetic algorithm
    Includes abstract classes for Crossover, Mutation and Selection operators
    """
    def __init__(self,
                 minimisation_objective_func: Callable[[np.ndarray], float],
                 named_constraint_funcs: dict,
                 constraint_penalty_weights: np.array,
                 pop_size: int,
                 elitism_size: int,
                 mutation_rate: float,
                 decision_var_shape: (int, int),
                 random_init_func: Callable[[int, int, int], np.ndarray]
                 ):
        """
        Initialise the base Genetic Algorithm class
        @param minimisation_objective_func: function to be minimised that takes np.array input
        @param named_constraint_funcs: dictionary of named constraint functions that are added as penalty terms
        @param constraint_penalty_weights: vector of objective function penalty weights for constraint terms
        @param pop_size: number of individual solutions in GA population
        @param elitism_size: number of top solutions carried forward to next generation
        @param mutation_rate: the proportion of genes undergoing mutation
        @param decision_var_shape: the shape of the decision variable matrix
        @param random_init_func: function for randomly initialising the GA population
        """
        self.minimisation_objective_func = minimisation_objective_func
        self.constraint_funcs = list(named_constraint_funcs.values())
        self.named_constraints = named_constraint_funcs
        self.constraint_penalty_weights = constraint_penalty_weights
        self.k = len(constraint_penalty_weights)
        assert len(self.constraint_funcs) == self.k
        self.decision_var_shape = decision_var_shape
        self.n = self.decision_var_shape[0]
        self.m = self.decision_var_shape[1]
        self.pop_size = pop_size if pop_size % 2 == 0 else pop_size + 1  # adjust pop_size to be even
        self.mutation_rate = mutation_rate
        self.max_constraint_violation_values = np.ones(len(constraint_penalty_weights))
        self.elitism_size = elitism_size
        assert self.elitism_size <= self.pop_size
        self.random_init_func = random_init_func

    def random_initialisation(self):
        """
        Randomly initialise the population of chromosomes
        :return: n x m x pop_size matrix of 0's and 1's
        """
        rand_pop = self.random_init_func(self.n, self.m, self.pop_size)
        return rand_pop

    def obj_func_with_penalty(self, solution: np.ndarray) -> float:
        """
        Calculate the sum of the main objective function and the
        total constraint function violations times their penalty weights
        :param solution: n x m solution matrix
        :return: fitness value
        """
        # calculate primary objective
        objective_val = self.minimisation_objective_func(solution)
        # calculate constraint violations
        constraint_violations = np.array([f(solution) for f in self.constraint_funcs])
        # update self.max_constraint_violation_values
        self.max_constraint_violation_values = np.where(self.max_constraint_violation_values < constraint_violations,
                                                        constraint_violations, self.max_constraint_violation_values)
        # calculate weighted sum of penalty violations
        penalty_val = np.dot(constraint_violations, self.constraint_penalty_weights)
        # return primary objective + weighted penalty
        return objective_val + penalty_val

    def obj_func(self, solution: np.ndarray) -> float:
        """
        Calculate the primary objective function value (no penalty)
        :param solution: n x m solution matrix
        :return: objective value
        """
        return self.minimisation_objective_func(solution)

    def calc_constraint_violations(self, solution: np.ndarray) -> np.array:
        """
        Calculate the constraint violations for each constraint function
        :param solution: n x m solution array
        :return: length k array of constraint violation values
        """
        # calculate constraint violations
        constraint_violations = np.array([f(solution) for f in self.constraint_funcs])
        return constraint_violations

    def is_feasible(self, solution: np.ndarray):
        return sum(self.calc_constraint_violations(solution)) == 0

    def sum_normalised_constraint_violations(self, constraint_violations: np.array):
        """
        Returns the min-max normalised constraint violations, using the current known maximum of each
        :param constraint_violations: length k output array of calc_constraint_violations
        :return: length k array of normalised constraint violations
        """
        # update self.max_constraint_violation_values
        self.max_constraint_violation_values = np.where(self.max_constraint_violation_values < constraint_violations,
                                                        constraint_violations, self.max_constraint_violation_values)
        return np.sum(constraint_violations / self.max_constraint_violation_values)

    @abstractmethod
    def crossover(self, parent1: np.array, parent2: np.array) -> (np.array, np.array):
        """
        Performs a crossover operation on two parent chromosomes to product 2 child solutions
        :param parent1: n x m matrix representing chromosome of parent solution 1
        :param parent2: n x m matrix representing chromosome of parent solution 2
        :return: n x m matrix representing child solution 1,
                 n x m matrix representing child solution 2
        """
        pass

    @abstractmethod
    def mutation(self, solution: np.ndarray, mutation_rate=None) -> np.array:
        """
        Randomly mutate genes in a solution with a low probability
        :param solution: n x m solution matrix to undergo mutation
        :param mutation_rate: the proportion of genes that will experience mutation at random
        :return: n x m mutated solution matrix
        """
        pass

    @abstractmethod
    def selection(self,
                  population: np.ndarray,
                  population_fitness: np.array) -> np.array:
        """
        Fitness biased selection of a single parent solution
        :param population: the n x m x pop_size matrix of solutions
        :param population_fitness: the pop_size length array of solution fitnesses
        :return: n x m array parent solution
        """
        pass

    def evaluate_fitness(self, population: np.ndarray) -> np.array:
        """
        Evaluate the fitness of the entire population of solutions
        :param population: n x m x pop_size matrix of solutions
        :return: pop_size length array of fitness values
        """
        fitness = [self.obj_func_with_penalty(population[:, :, i]) for i in range(self.pop_size)]
        return fitness

    # evaluate the fitness of the population, select parents and generate children
    # return children and their fitness
    def generate_children(self,
                          population: np.ndarray,
                          pop_fitness: np.array
                          ) -> (np.ndarray, np.array):
        """
        Generate pop_size children using parents selected from the population with a fitness bias
        :param population: n x m x pop_size matrix of solutions
        :param pop_fitness: pop_size length array of solution fitness
        :return: n x m x pop_size matrix of child solutions,
                 pop_size length array of child solution fitnesses
        """
        # select pop_size/2 groups of 2 parents
        parents = [(self.selection(population, pop_fitness), self.selection(population, pop_fitness))
                   for _ in range(int(len(pop_fitness)/2))]
        # apply crossover
        children = [self.crossover(p[0], p[1]) for p in parents]
        children = list(sum(children, ()))
        # apply mutation
        children = [self.mutation(c) for c in children]
        child_fitness = np.array([self.obj_func_with_penalty(c) for c in children])
        children = np.dstack(children)
        return children, child_fitness

    def survival(self, population, population_fitness, children, child_fitness) -> (np.ndarray, np.array):
        """
        Elitist survival of pop_size solutions from the combined current & child populations
        The top self.elitism_size solutions are kept from one generation to the next
        :param population: population of pop_size solutions
        :param population_fitness: pop_size length array of population solution fitnesses
        :param children: generated population of pop_size children
        :param child_fitness: pop_size length array of children solution fitnesses
        :return: new n x m x pop_size population to survive to the next generation,
                 pop_size length array of new solution fitnesses
        """
        combined_population = np.concatenate((population, children), axis=2)
        combined_fitness = np.concatenate((population_fitness, child_fitness))
        elite_fitness_idx = np.array([y for x, y in
                                      sorted([(f, i) for f, i in zip(combined_fitness, range(len(combined_fitness)))],
                                             key=lambda x: x[0], reverse=False)[:self.elitism_size]])
        other_idx = [i for i in range(self.pop_size) if i not in elite_fitness_idx]
        remaining = self.pop_size - self.elitism_size
        other_idx = np.random.choice(other_idx, size=remaining, replace=False)
        new_population = np.dstack([combined_population[:, :, e_i] for e_i in elite_fitness_idx] + \
                                   [combined_population[:, :, o_i] for o_i in other_idx])
        new_population_fitness = np.array([combined_fitness[e_i] for e_i in elite_fitness_idx] + \
                                          [combined_fitness[o_i] for o_i in other_idx])
        return new_population, new_population_fitness

    def solution_performance_report(self, solution: np.ndarray):
        """
        Prints a performance report for a solution,
        showing objective function performance and constraint violations
        :param solution: n x m solution matrix
        """
        if self.is_feasible(solution):
            print("Solution is feasible")
        else:
            print("Solution is infeasible")
        print("Solution minimisation objective value: " + str(int(self.obj_func(solution))))
        print("Solution fitness value: " + str(np.round(self.obj_func_with_penalty(solution), 2)))
        for constr_name, constr_func in self.named_constraints.items():
            print(constr_name + " violation: " + str(np.round(constr_func(solution), 2)))
        print("Solution:")
        print(solution)
