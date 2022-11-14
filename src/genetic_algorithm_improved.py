# Improved Genetic Algorithm class
import numpy as np
from problem import *
import random
from typing import Callable
from base import BaseGA


class ImprovedGA(BaseGA):
    """
    Improved Genetic Algorithm class
    Stateless - does not store solution populations internally
    Initialised with objective & constraint functions, penalty weights and optimisation parameters
    Adds random solution injection, solution repair, local search with swaps
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
                 random_injection_probability: float = 0.01,
                 repair_probability: float = 0.05,
                 include_design_knowledge: bool = False,
                 swap_prob: float = 0.05,
                 num_swaps: int = 10,
                 ):
        """
        Initialise the Genetic Algorithm class (improved)
        @param minimisation_objective_func: function to be minimised that takes np.array input
        @param named_constraint_funcs: dictionary of named constraint functions that are added as penalty terms
        @param constraint_penalty_weights: vector of objective function penalty weights for constraint terms
        @param pop_size: number of individual solutions in GA population
        @param elitism_size: number of top solutions carried forward to next generation
        @param mutation_rate: the proportion of genes undergoing mutation
        @param decision_var_shape: the shape of the decision variable matrix
        @param random_init_func: function for randomly initialising the GA population
        @param random_injection_probability: probability of adding a randomly generated solution per generation
        @param repair_probability: probability of repairing a solution
        @param include_design_knowledge: if True include solutions informed by design knowledge in init population
        @param swap_prob: probability of conducting local search per child solution
        @param num_swaps: number of cell values to swap for local search
        """
        super().__init__(minimisation_objective_func, named_constraint_funcs, constraint_penalty_weights,
                         pop_size, elitism_size, mutation_rate, decision_var_shape, random_init_func)
        self.const_row1 = np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1])
        self.const_row2 = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1])
        self.const_row3 = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1])
        self.const_row4 = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
        self.const_row5 = np.array([1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1])
        self.random_injection_probability = random_injection_probability
        self.repair_probability = repair_probability
        self.include_design_knowledge = include_design_knowledge
        self.swap_prob = swap_prob
        self.num_swaps = num_swaps

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

    def repair(self, solution: np.ndarray):
        for i in range(self.n):
            array_idx = random.randint(0, self.m-5)
            while array_idx < self.m:
                if solution[i, array_idx] == 1:
                    ones_count = 1
                    try:
                        while solution[i, array_idx + 1] == 1 and ones_count < 5:
                            ones_count += 1
                            array_idx += 1
                        array_idx += 1
                        if ones_count == 5:
                            solution[i, array_idx + random.randint(-4, 0)] = 1
                            array_idx = self.m
                    except IndexError:
                        array_idx += 1
                else:
                    array_idx += 1
        return solution

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> (np.array, np.array):
        """
        Performs a crossover operation on two parent chromosomes to product 2 child solutions
        :param parent1: n x m matrix representing chromosome of parent solution 1
        :param parent2: n x m matrix representing chromosome of parent solution 2
        :return: n x m matrix representing child solution 1,
                 n x m matrix representing child solution 2
        """
        y_cross = random.randint(1, len(parent1) - 1)
        x_cross = random.randint(1, parent1.shape[1] - 1)
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)
        child1[:y_cross, :x_cross] = parent2[:y_cross, :x_cross]
        child2[:y_cross, :x_cross] = parent1[:y_cross, :x_cross]
        child1[y_cross:, x_cross:] = parent2[y_cross:, x_cross:]
        child2[y_cross:, x_cross:] = parent1[y_cross:, x_cross:]
        return child1, child2

    def mutation(self, solution: np.ndarray, mutation_rate=None) -> np.array:
        """
        Randomly mutate genes in a solution with a low probability
        :param solution: n x m solution matrix to undergo mutation
        :param mutation_rate: the proportion of genes that will experience mutation at random
        :return: n x m mutated solution matrix
        """
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        rand_matrix = np.random.rand(self.n, self.m)
        mutated = np.where(rand_matrix < mutation_rate, random.randint(0, 1), solution)
        return mutated

    @staticmethod
    def selection_rw(population: np.ndarray,
                     population_fitness: np.array) -> np.array:
        """
        Fitness biased selection of a single parent solution (Roulette Wheel)
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

    def swap(self, solution: np.ndarray):
        """
        Local search that swaps values between cells in decision variable matrix
        @param solution: decision variable matrix
        @return: solution with values at a random number of indices swapped
        """
        num_swaps = random.randint(1, self.num_swaps)
        for i in range(num_swaps):
            col1 = random.randint(0, self.m-1)
            row1 = random.randint(0, self.n-1)
            col2 = random.randint(0, self.m - 1)
            row2 = random.randint(0, self.n - 1)
            temp = solution[row1, col1]
            solution[row1, col2] = solution[row2, col2]
            solution[row2, col2] = temp
        return solution

    def select_random_parent_indices(self) -> (int, int):
        idx1 = random.randint(0, self.pop_size - 1)
        idx2 = idx1
        while idx2 == idx1:
            idx2 = random.randint(0, self.pop_size - 1)
        return idx1, idx2

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
        idx1, idx2 = self.select_random_parent_indices()
        if population_fitness[idx1] > population_fitness[idx2]:
            return population[:, :, idx1]
        else:
            return population[:, :, idx2]

    def evaluate_fitness(self, population: np.ndarray) -> np.array:
        """
        Evaluate the fitness of the entire population of solutions
        :param population: n x m x pop_size matrix of solutions
        :return: pop_size length array of fitness values
        """
        population_fitness = \
            np.array([self.obj_func_with_penalty(population[:, :, z]) for z in range(self.pop_size)])
        return population_fitness

    # evaluate the fitness of the population, select parents and generate children
    # return children and their fitness
    def generate_children(self,
                          population: np.ndarray,
                          pop_fitness: np.array
                          ) -> (np.ndarray, np.array):
        """
        New version to generate pop_size children using parents selected from the population with a fitness bias
        Adds random injection, solution repair and local search
        :param population: n x m x pop_size matrix of solutions
        :param pop_fitness: pop_size length array of solution fitness
        :return: n x m x pop_size matrix of child solutions,
                 pop_size length array of child solution fitnesses
        """
        # select pop_size/2 groups of 2 parents
        if np.random.random() < self.random_injection_probability:
            random_solutions = (np.round(np.random.rand(self.n, self.m)), np.round(np.random.rand(self.n, self.m)))
            parents = [(self.selection(population, pop_fitness), self.selection(population, pop_fitness))
                       for _ in range(int(self.pop_size / 2)-1)]
            parents.append(random_solutions)
        else:
            parents = [(self.selection(population, pop_fitness), self.selection(population, pop_fitness))
                       for _ in range(int(self.pop_size / 2))]
        # apply crossover
        children = [self.crossover(p[0], p[1]) for p in parents]
        children = list(sum(children, ()))
        # apply mutation
        children = [self.mutation(c) for c in children]
        children = [child if random.random() > self.repair_probability else self.repair(child) for child in children]
        children = [child if random.random() > self.swap_prob else self.swap(child) for child in children]
        child_fitness = np.array([self.obj_func_with_penalty(c) for c in children])
        children = np.dstack(children)
        return children, child_fitness
