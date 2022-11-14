from src.problem import *
from src.genetic_algorithm_EDIT import *
import pytest


test_num_shifts = 7
test_num_junior_elves = 2
test_num_senior_elves = 1
test_workshop_capacities = np.array([1, 2, 1, 1, 1, 2, 1])
test_shifts = np.array([[1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0]])
toys_made = 10 * 10000 * 12 + 15000 * 12
cookies_earned = 10 * 3 * 12 + 4 * 12
test_weights = np.array([1, 1, 1, 1])


def test_calculate_toy_shortfall():
    assert calculate_toy_shortfall(elf_shift_vars=test_shifts,
                                   num_junior_elves=test_num_junior_elves) == 500000000 - toys_made


def test_workshop_capacity_violation():
    assert workshop_capacity_violation(elf_shift_vars=test_shifts, capacities=test_workshop_capacities) == 4


def test_consecutive_shift_violation():
    assert consecutive_shift_violation(elf_shift_vars=test_shifts, max_consecutive=4, n_shifts=test_num_shifts) == 2.0


def test_elf_max_min_ratio_violation():
    assert elf_max_min_ratio_violation(elf_shift_vars=test_shifts) == 3.5


def test_calculate_cookie_wages():
    assert calculate_cookie_wages(elf_shift_vars=test_shifts, num_junior_elves=2) == cookies_earned


def test_random_shifts_initialisation():
    assert random_shifts_initialisation(num_elves=3, num_shifts=7).shape == (3, 7)
    assert random_shifts_initialisation(num_elves=3, num_shifts=7, pop_size=10).shape == (3, 7, 10)


def wrap_calculate_toy_shortfall(test_shifts):
    return calculate_toy_shortfall(elf_shift_vars=test_shifts, num_junior_elves=test_num_junior_elves)


def wrap_workshop_capacity_violation(test_shifts):
    return workshop_capacity_violation(elf_shift_vars=test_shifts, capacities=test_workshop_capacities)


def wrap_calculate_cookie_wages(test_shifts):
    return calculate_cookie_wages(elf_shift_vars=test_shifts, num_junior_elves=test_num_junior_elves)


def test_objective_function_with_penalty():
    named_constraints = {"Toy shortfall": wrap_calculate_toy_shortfall,
                         "Workshop capacity": wrap_workshop_capacity_violation,
                         "Consecutive shifts": consecutive_shift_violation,
                         "Max/min work ratios": elf_max_min_ratio_violation,
                         }

    ga = ImprovedGA(minimisation_objective_func=wrap_calculate_cookie_wages,
                    named_constraint_funcs=named_constraints,
                    constraint_penalty_weights=test_weights,
                    pop_size=10,
                    elitism_size=10,
                    mutation_rate=0.01,
                    decision_var_shape=(3, 7),
                    random_init_func=random_shifts_initialisation)
    assert ga.obj_func_with_penalty(solution=test_shifts) == (cookies_earned + (500000000 - toys_made) + 7 + 2 + 3.5)


# test_calculate_toy_shortfall()
# test_workshop_capacity_violation()
# test_consecutive_shift_violation()
# test_elf_max_min_ratio_violation()
# test_calculate_cookie_wages()
# test_random_shifts_initialisation()
# test_objective_function_with_penalty()
