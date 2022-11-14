"""
Santa's Workshop Staffing Problem
Defines the constants and functions needed for the SWSP
introduced in `ga-challenge/genetic_algorithm_challenge.ipynb`
"""
import numpy as np
from numba import njit

# DO NOT EDIT BELOW vvv #######################################################

# define known problem constants
num_junior_elves = 150
num_senior_elves = 50
num_shifts = 28
hours_per_shift = 12
junior_toys_per_shift = 10000 * 12
senior_toys_per_shift = 15000 * 12
junior_cookie_rate = 3
senior_cookie_rate = 4

shift_workshop_capacities = np.append(np.ones(14)*150, np.ones(14)*140)
max_min_ratio = 1.5
max_consecutive_shifts = 4
num_toys = 500000000

# sample decision variable chromosomes
# 0 indicates an elf is NOT working that shift
# 1 indicates an elf IS working that shift
# decision variable matrix with a row for every elf, and a column for every shift.
# By construction, the first num_junior_elves rows are for junior elves
np.random.seed(2)
random_elf_shift_vars = np.round(np.random.rand(num_junior_elves + num_senior_elves, num_shifts))
all_ones_shift_vars = np.ones((num_junior_elves + num_senior_elves, num_shifts))
all_zeros_shift_vars = np.zeros((num_junior_elves + num_senior_elves, num_shifts))
all_ones_shift_vars_bar_one = np.copy(all_ones_shift_vars)
all_ones_shift_vars_bar_one[0, :] = np.zeros(num_shifts)


# helper function
@njit
def count_consecutive_ones(arr, cols=num_shifts):
    """
    Helper function to calculate number of 1's in each group of consecutive 1's in an individual elf's shift
    @param arr: row of decision matrix representing an individual elf's schedule
    @param cols: the total number of shifts in a single schedule
    @return: array of length int(cols/2) where each element is the number of 1's per group of consecutive 1's
    """
    length = int(cols/2)
    groups_of_ones = np.zeros(length)
    group_idx = 0
    array_idx = 0
    while array_idx < cols:
        if arr[array_idx] == 1:
            ones_count = 1
            if array_idx+1 < cols:
                while arr[array_idx+1] == 1:
                    ones_count += 1
                    if array_idx+2 < cols:
                        array_idx += 1
                    else:
                        break
                groups_of_ones[group_idx] = ones_count
                group_idx += 1
                array_idx += 1
            else:
                if group_idx < length:
                    groups_of_ones[group_idx] = ones_count
                    array_idx += 1
                else:
                    array_idx += 1
        else:
            array_idx += 1
    return groups_of_ones


# constraint functions
def calculate_toy_shortfall(elf_shift_vars,
                            num_junior_elves=num_junior_elves,
                            junior_toys_per_shift=junior_toys_per_shift,
                            senior_toys_per_shift=senior_toys_per_shift,
                            toys_needed=num_toys,
                            calculate_produced=False,
                            ):
    """
    Calculates the total toy shortfall relative to toys_needed
    Addresses constraint: Toy production requirements (>= 500,000,000)
    @param elf_shift_vars: num_elves x num_shifts binary matrix of decision variables
    @param num_junior_elves: the total number of junior elves
    @param junior_toys_per_shift: the number of toys a junior elf can make in a shift
    @param senior_toys_per_shift: the number of toys a senior elf can make in a shift
    @param toys_needed: the total number of toys to be manufactured over the planning horizon
    @param calculate_produced: if True return total num toys produced instead
    @return: shortfall of toys produced versus toys needed for a given solution, or zero if feasible
    """
    junior_toys_produced = np.sum(elf_shift_vars[:num_junior_elves, :] * junior_toys_per_shift)
    senior_toys_produced = np.sum(elf_shift_vars[num_junior_elves:, :] * senior_toys_per_shift)
    if calculate_produced:
        return junior_toys_produced + senior_toys_produced
    else:
        return max(toys_needed - (junior_toys_produced + senior_toys_produced), 0)


def workshop_capacity_violation(elf_shift_vars, capacities=shift_workshop_capacities):
    """
    Addresses constraint:  There is only space for 150 elves at a time in Santa’s workshop during the first week,
    and 140 elves during the last week (Santa’s enormous toy sack displaces 10 elves)
    @param elf_shift_vars: num_elves x num_shifts binary matrix of decision variables
    @param capacities: num_shifts length array holding the workshop capacity per shift
    @return: sum total of workshop capacity violations over all shifts for a given solution, or zero if feasible
    """
    elfs_per_shift = np.sum(elf_shift_vars, axis=0)
    capacity_violations = np.where(elfs_per_shift <= capacities, 0,
                                   elfs_per_shift - capacities)
    return np.sum(capacity_violations)


def consecutive_shift_violation(elf_shift_vars, max_consecutive=max_consecutive_shifts, n_shifts=num_shifts):
    """
    Addresses constraint: No individual elf may work more than 48 hours (i.e. 4 12-hour shifts) in a row
    @param elf_shift_vars: num_elves x num_shifts binary matrix of decision variables
    @param max_consecutive: maximum number of consecutive shifts an elf may work
    @param n_shifts: total number of shifts to schedule
    @return: the sum total of max consecutive shifts violations for a given solution, or zero if feasible
    """
    # count groups of consecutive shifts per elf
    count_groups = np.apply_along_axis(count_consecutive_ones, 1, elf_shift_vars, n_shifts).flatten()
    # filter out groups less than or equal to 4 consecutive shifts
    count_groups = np.where(count_groups <= max_consecutive, 0, count_groups-max_consecutive)
    # sum the total constraint violation
    excess_shifts = np.sum(count_groups)
    return excess_shifts


def elf_max_min_ratio_violation(elf_shift_vars, max_min_ratio=max_min_ratio) -> float:
    """
    Addresses constraint: The most hard-working elf must work a maximum of 50% more hours than the
    least hard-working elf (in total)
    @param elf_shift_vars: num_elves x num_shifts binary matrix of decision variables
    @param max_min_ratio: the maximum ratio of the most number of shifts worked to the least
    @return: violation of max_min_ratio for a given solution, or zero if feasible
    """
    elfs_per_shift = np.sum(elf_shift_vars, axis=1)
    max_shifts = np.max(elfs_per_shift)
    min_shifts = max(np.min(elfs_per_shift), 1)  # prevent division by zero
    return max(0, max_shifts / min_shifts - max_min_ratio)


# problem-specific objective and penalty functions
def calculate_cookie_wages(elf_shift_vars,
                           num_junior_elves=num_junior_elves,
                           junior_cookies_per_hour=junior_cookie_rate,
                           senior_cookies_per_hour=senior_cookie_rate,
                           hours_per_shift=hours_per_shift,
                           ):
    """
    Objective function that calculates the total wages in units of cookies
    Junior elf wages = 3 cookies per hour
    Senior elf wages = 4 cookies per hour
    @param elf_shift_vars: num_elves x num_shifts binary matrix of decision variables
    @param num_junior_elves: the total number of junior elves
    @param junior_cookies_per_hour: junior elf cookie wages rate per hour
    @param senior_cookies_per_hour: senior elf cookie wages rate per hour
    @param hours_per_shift: total number of hours per shift
    @return: sum total of junior and senior elf cookie wages for a given solution
    """
    junior_cookie_wages = np.sum(elf_shift_vars[:num_junior_elves, :] * hours_per_shift * junior_cookies_per_hour)
    senior_cookie_wages = np.sum(elf_shift_vars[num_junior_elves:, :] * hours_per_shift * senior_cookies_per_hour)
    return junior_cookie_wages + senior_cookie_wages


def random_shifts_initialisation(num_elves: int, num_shifts: int, pop_size:int=None):
    """
    Creates a randomised solution or population of solutions for the shift scheduling problem
    @param num_elves: number of rows representing elves
    @param num_shifts: number of columns representing shifts
    @param pop_size: population size for genetic algorithm
    @return: num_elves x num_shifts x pop_size binary matrix of uniformly random solutions
    """
    if pop_size is not None:
        return np.round(np.random.rand(num_elves, num_shifts, pop_size))
    else:
        return np.round(np.random.rand(num_elves, num_shifts))

# DO NOT EDIT ABOVE ^^^ #######################################################

# best known objective value: 144144 cookies
# best known solution:
# [[1,1,0,1,0,0,0,0,0,1,0,1,1,1,1,0,1,0,1,1,1,0,1,1,0,0,0,1],
#  [0,0,1,1,0,1,0,0,1,1,0,1,0,1,1,0,1,1,0,1,1,1,0,1,0,1,1,1],
#  [1,0,1,0,1,1,1,0,1,0,1,1,0,1,1,1,0,1,1,1,1,0,1,1,0,1,1,1],
#  [1,1,0,0,1,1,1,1,0,0,1,1,0,1,1,0,1,1,0,0,1,1,0,1,1,1,1,0],
#  [1,1,0,1,0,1,1,1,1,0,1,0,1,0,0,1,1,0,1,1,0,1,1,1,1,0,1,1],
#  [1,0,1,0,1,0,1,1,1,1,0,0,1,1,1,1,0,0,1,0,0,0,1,0,1,1,1,1],
#  [0,1,1,0,1,0,1,1,1,0,1,1,0,1,0,0,0,0,0,1,1,0,1,0,1,1,1,1],
#  [1,0,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,0,0],
#  [1,0,0,1,0,1,0,0,1,1,0,1,0,0,1,0,1,1,0,1,1,1,1,0,1,0,0,1],
#  [1,1,1,0,1,1,1,0,0,1,1,1,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1],
#  [0,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,1,0,0,1,0,1,1,1,0,1],
#  [1,1,1,1,0,1,1,1,1,0,0,1,0,0,1,1,0,0,1,0,0,1,0,1,1,1,0,0],
#  [1,1,0,1,1,0,0,0,0,1,1,1,0,1,1,1,0,0,1,0,0,1,0,1,1,1,0,0],
#  [1,1,0,1,0,1,0,1,1,1,0,0,0,0,1,1,0,1,1,1,1,0,1,1,0,1,0,1],
#  [1,1,1,0,1,1,1,1,0,1,0,1,0,0,0,1,1,1,0,1,0,1,1,1,1,0,1,1],
#  [0,1,1,0,1,1,0,1,1,0,1,0,1,1,1,1,0,1,1,0,1,1,1,0,0,0,1,0],
#  [1,0,1,1,1,1,0,1,1,1,1,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,1,1],
#  [0,1,1,0,1,1,0,1,1,0,1,1,1,0,1,1,1,1,0,1,1,0,0,1,0,1,1,0],
#  [1,1,1,1,0,1,1,1,0,1,1,0,0,1,1,1,0,1,0,1,1,1,1,0,1,1,0,1],
#  [1,1,1,1,0,0,1,1,0,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,0,1,1,1],
#  [0,1,0,1,1,1,0,1,1,1,0,1,1,0,1,0,1,1,1,0,1,0,0,0,1,0,1,0],
#  [1,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,0,1,0,0],
#  [0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,0,1,1,0,1,1,0,1,1,1,1],
#  [1,1,1,0,0,1,0,1,1,1,0,0,1,1,0,1,0,1,1,1,1,0,0,0,1,1,1,1],
#  [1,1,0,0,1,1,1,0,1,1,1,1,0,0,1,1,1,1,0,0,0,1,1,1,0,1,1,1],
#  [1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,0,0,0,1,1,1,0,1,1,1,0],
#  [1,0,1,1,1,1,0,1,1,1,1,0,1,0,0,0,1,1,0,1,1,1,1,0,0,0,1,1],
#  [0,1,0,0,1,0,1,1,1,0,1,0,1,0,1,1,1,1,0,1,0,1,0,1,1,1,0,1],
#  [1,0,1,0,1,1,1,1,0,1,1,1,1,0,0,1,0,1,1,1,1,0,0,1,1,0,0,1],
#  [0,0,1,0,0,1,0,1,1,1,0,1,1,0,1,1,0,1,0,1,1,1,0,1,0,1,1,0],
#  [1,0,1,1,1,0,1,1,0,0,0,1,1,1,0,1,0,1,0,1,0,0,1,1,1,0,1,0],
#  [1,1,0,0,0,0,1,1,1,1,0,0,1,1,1,1,0,1,0,0,1,1,1,0,1,1,1,1],
#  [0,1,0,1,0,1,0,1,1,0,1,1,1,0,1,1,0,1,0,1,1,0,0,1,1,0,1,1],
#  [1,1,1,0,1,1,1,0,0,1,0,1,1,1,1,0,0,1,0,1,1,1,1,0,1,0,1,1],
#  [0,0,1,0,1,1,1,0,1,1,1,1,0,1,0,1,0,1,1,1,0,1,1,1,1,0,0,0],
#  [1,0,1,1,1,1,0,1,1,0,1,1,0,1,1,1,1,0,0,1,1,0,1,0,1,1,1,1],
#  [1,1,1,0,0,0,1,1,1,0,1,0,1,1,0,1,1,1,0,1,1,0,0,0,1,1,1,1],
#  [0,1,1,1,0,1,1,0,0,1,0,1,1,0,0,1,1,0,0,1,0,0,1,1,1,1,0,0],
#  [1,0,1,1,1,1,0,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,0,1,0,1,1,1],
#  [1,0,1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,0,1,0,0,1,0,1,0,1,1,0],
#  [1,0,1,0,1,1,1,0,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1],
#  [1,1,0,1,0,0,1,0,0,0,1,1,1,0,1,1,1,0,0,0,1,1,1,0,1,1,1,1],
#  [1,1,0,0,1,1,1,1,0,1,0,0,1,1,1,0,0,1,1,1,1,0,1,0,1,1,1,0],
#  [1,0,1,1,1,0,1,0,1,0,0,1,1,0,0,1,0,0,1,1,1,1,0,0,1,1,1,1],
#  [1,1,1,0,1,1,1,1,0,0,1,1,1,0,0,1,1,1,1,0,0,0,0,1,1,1,0,1],
#  [0,0,1,1,1,1,0,1,0,0,1,1,1,0,1,0,1,1,0,1,0,1,1,1,0,1,1,1],
#  [1,1,1,0,1,1,0,1,1,1,1,0,0,0,1,1,1,1,0,1,1,0,1,1,1,0,1,1],
#  [0,1,1,1,1,0,0,1,1,1,0,1,1,0,1,1,0,1,1,1,0,1,1,1,0,0,1,1],
#  [0,1,1,1,0,1,1,0,1,0,1,0,1,1,1,1,0,1,1,1,0,1,1,0,1,1,1,1],
#  [1,1,0,1,1,1,0,1,0,1,1,1,0,0,1,1,1,0,1,0,1,0,1,0,1,1,1,0],
#  [1,1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,1,0,0,1,0,1,1,1,0,0,1,1],
#  [0,0,0,1,0,1,0,1,1,0,1,1,0,1,1,1,1,0,1,1,0,1,0,1,1,1,0,1],
#  [1,1,0,0,1,0,1,1,1,0,1,1,0,0,1,1,1,0,0,1,1,1,0,1,1,1,0,0],
#  [1,1,1,1,0,1,1,0,1,1,0,1,1,1,0,0,1,1,0,1,1,1,1,0,1,1,1,1],
#  [1,1,1,0,1,1,1,1,0,1,1,0,0,1,0,1,1,0,0,1,1,0,1,1,0,1,0,1],
#  [1,1,0,0,1,0,1,0,0,0,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,0,1],
#  [1,1,1,1,0,0,1,1,1,1,0,0,0,1,0,1,1,1,0,1,1,1,1,0,1,1,0,0],
#  [1,0,1,1,0,1,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,0,1,1,1,1,0],
#  [1,1,0,1,1,1,0,1,1,0,0,1,1,0,1,1,1,0,1,1,1,1,0,1,1,0,0,1],
#  [1,0,1,1,1,1,0,1,0,1,0,0,1,0,1,1,1,0,1,1,0,0,1,1,0,1,1,1],
#  [1,0,1,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,0,1,0,1,1,0,0,1,0,0],
#  [0,1,1,0,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1,0],
#  [1,0,1,1,0,1,0,1,1,1,1,0,1,0,1,1,0,1,1,1,0,1,1,1,0,1,0,1],
#  [0,1,1,1,0,1,1,0,0,1,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,0,1,1],
#  [1,1,1,0,0,1,0,1,0,1,0,0,0,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1],
#  [1,1,0,1,1,0,1,1,1,0,0,1,1,1,1,0,1,0,0,1,1,1,0,1,0,1,0,1],
#  [1,1,0,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,0,1,1,1,1,0,1,0,0,1],
#  [1,0,1,1,1,0,1,0,1,1,1,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,1,1],
#  [0,0,1,0,1,0,0,1,1,1,0,1,0,1,1,1,0,1,1,0,1,1,0,1,1,1,1,0],
#  [1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,1,0,0,1,1,0,1,0,0,0,1,1,1],
#  [1,0,1,1,1,0,1,1,1,0,1,0,0,0,1,0,1,1,1,1,0,1,0,1,1,1,0,0],
#  [0,1,0,1,0,1,1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,0,1,1,1,1,0],
#  [1,1,1,0,1,1,0,1,1,1,1,0,1,1,1,0,1,1,0,1,0,0,1,1,0,1,0,0],
#  [1,1,1,1,0,1,1,1,0,1,0,1,1,0,1,1,1,1,0,1,0,1,1,0,1,1,1,1],
#  [1,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,0,0,1,1,1,0,1,1,1,0,1,1],
#  [1,0,1,1,1,1,0,0,0,0,1,1,0,1,1,0,1,1,0,0,1,0,1,1,0,1,1,1],
#  [0,1,1,1,0,0,0,1,0,0,1,1,0,1,0,1,1,1,1,0,1,1,1,1,0,0,1,0],
#  [1,0,1,1,1,0,1,1,1,1,0,0,0,0,1,1,0,0,0,1,1,0,1,1,1,1,0,1],
#  [1,0,1,1,1,0,0,1,1,0,1,1,1,1,0,1,0,0,0,1,0,1,1,1,1,0,1,0],
#  [1,1,0,1,0,1,0,1,1,0,1,0,1,1,1,1,0,0,1,0,1,1,0,1,1,0,1,1],
#  [1,1,1,0,1,0,1,1,0,0,1,1,1,0,1,1,1,0,1,0,1,0,0,1,1,0,1,1],
#  [1,0,0,1,1,0,1,0,1,1,0,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,0,1],
#  [1,0,1,1,1,0,1,1,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,0,1,1,1,1],
#  [1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,1,0,0,1,0,1,1,0,1,1,1,0,1],
#  [1,1,0,0,0,1,1,1,0,1,1,1,0,0,1,0,1,0,1,0,1,1,1,1,0,1,1,1],
#  [1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,0,1,0,1,1,1,1,0],
#  [1,1,0,0,1,0,1,1,0,1,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1,0,1,1],
#  [1,0,1,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,1,1,1],
#  [1,0,0,0,0,1,0,1,1,0,1,1,1,0,1,0,1,1,0,1,0,0,1,0,1,1,1,1],
#  [1,1,1,0,1,1,0,1,1,0,0,1,0,1,1,1,0,0,1,1,1,0,1,0,1,1,1,1],
#  [1,0,1,1,0,1,0,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1,0,1,1,1,0,1],
#  [0,1,0,1,1,0,1,1,0,1,0,0,1,1,1,1,0,0,0,1,0,1,0,1,1,0,1,0],
#  [1,0,0,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1],
#  [0,0,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,0],
#  [1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,1,0,1,1,1,1],
#  [1,0,0,0,1,0,1,0,1,1,1,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,1,0],
#  [1,1,1,0,0,1,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,0,1,1,0,0,1,0],
#  [0,0,1,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1],
#  [1,0,1,1,1,1,0,1,0,1,0,0,1,0,0,1,1,1,0,1,1,0,0,1,1,0,1,1],
#  [1,0,1,1,1,1,0,1,1,0,1,1,1,1,0,0,1,1,1,0,1,1,1,0,1,1,0,0],
#  [0,1,0,0,1,0,1,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,0,1,1,0,1],
#  [0,0,1,0,1,1,0,1,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,1,0,1,1,0],
#  [1,1,1,1,0,0,1,0,1,1,1,1,0,1,0,1,1,0,0,1,1,1,0,1,0,1,1,1],
#  [1,0,1,0,1,1,0,0,1,0,1,1,1,0,0,0,1,1,1,1,0,1,1,1,0,1,0,1],
#  [1,1,1,1,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,0,1,0,0,1,0,1,1,1],
#  [1,1,0,1,0,1,0,1,1,1,1,0,1,1,1,0,1,0,1,1,0,0,1,1,0,1,1,1],
#  [1,0,1,1,1,0,1,1,1,1,0,1,1,0,1,0,0,1,0,1,1,0,0,1,0,1,0,1],
#  [1,0,1,1,1,1,0,1,1,1,1,0,1,0,1,0,1,1,0,1,1,0,1,1,0,1,1,0],
#  [1,1,1,0,1,1,1,0,1,1,1,1,0,0,1,0,1,1,1,1,0,1,1,1,0,1,1,1],
#  [1,1,1,0,1,0,1,1,0,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,1,0,1],
#  [1,1,0,1,0,1,0,1,1,1,1,0,0,0,0,1,1,0,1,1,1,1,0,1,1,1,1,0],
#  [1,1,1,0,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,0,1,1,0,1],
#  [1,1,0,1,1,0,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,1,0,0,1],
#  [0,1,0,1,1,1,0,0,1,0,1,1,1,0,1,1,1,1,0,1,1,0,1,1,1,0,0,1],
#  [1,1,1,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,1,1,1,0,1,1,1,0,1,0],
#  [1,1,1,0,1,0,1,1,1,0,1,1,0,0,1,1,0,0,1,0,1,1,1,1,0,0,0,0],
#  [0,1,1,1,0,1,0,0,1,0,0,1,1,1,0,1,0,1,0,1,1,0,0,1,0,1,1,1],
#  [0,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,1,0,0,0,0,1,1,1,1,0,0,1],
#  [1,0,1,1,1,1,0,1,1,0,1,1,1,1,0,1,1,1,0,1,0,0,1,0,1,1,0,0],
#  [1,0,0,1,1,0,1,1,1,1,0,1,1,0,0,1,1,1,0,1,1,0,1,1,1,1,0,1],
#  [0,1,0,0,1,0,0,1,1,1,1,0,0,1,1,1,0,1,1,0,1,0,0,1,1,0,1,1],
#  [0,0,1,1,1,1,0,1,0,1,1,0,0,1,1,0,1,1,0,0,1,0,1,1,1,0,1,1],
#  [0,1,1,1,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,0,1,0,1,0,1,1,1,0],
#  [0,0,1,1,1,1,0,1,0,1,1,0,0,1,0,1,1,1,0,1,0,1,1,0,1,0,1,1],
#  [0,1,1,1,1,0,0,1,0,0,1,1,1,0,1,1,0,1,1,0,1,0,1,1,1,1,0,1],
#  [0,1,0,1,1,0,0,0,0,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,0,1,1,0],
#  [1,1,1,0,0,1,0,1,1,1,0,1,0,1,1,1,0,1,1,0,0,1,1,0,1,1,0,1],
#  [1,1,0,1,1,0,0,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0],
#  [1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,0,1,1,0,0,1],
#  [0,1,1,1,0,1,1,1,0,0,0,1,1,0,0,1,0,1,0,1,0,1,1,0,1,1,1,0],
#  [1,1,1,0,0,1,1,1,1,0,1,0,1,1,0,1,0,1,1,1,1,0,0,1,1,0,1,0],
#  [1,0,1,0,1,1,1,0,1,1,0,1,1,0,0,0,1,1,0,0,0,0,1,1,0,1,1,1],
#  [0,1,1,1,0,1,1,1,1,0,1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,1,0,1],
#  [1,1,1,1,0,1,1,1,0,0,1,0,0,1,0,1,0,0,1,1,1,0,1,1,0,0,1,1],
#  [1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,0,0,1,1,0,0,1,1,1],
#  [0,1,0,1,1,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,1,1,0,1,0,0,1,1],
#  [1,1,1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,1,1,1,0,1,0,0,1,0,0],
#  [1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,0,1,1,1,0,1,1,0,1,0,1,0,1],
#  [1,1,1,0,0,1,1,1,1,0,0,1,1,0,1,1,0,1,0,1,1,1,0,0,1,1,1,1],
#  [1,1,0,1,1,1,1,0,1,1,1,0,1,1,0,1,0,0,0,1,1,1,1,0,0,0,1,0],
#  [1,0,1,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,1,0,1,0,1,1,0,0],
#  [1,1,1,1,0,1,1,0,1,0,1,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,1,1],
#  [1,0,1,1,1,1,0,0,1,1,0,1,1,0,1,1,1,1,0,1,1,1,0,1,0,1,1,1],
#  [0,1,1,0,1,0,0,1,0,0,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,0,1,1],
#  [1,1,1,0,0,1,0,1,1,1,1,0,1,1,0,1,1,1,0,0,1,1,1,0,1,0,1,0],
#  [0,0,1,1,1,1,0,0,0,1,0,1,1,0,0,0,1,0,1,1,1,1,0,1,1,1,0,0],
#  [1,0,0,1,1,1,1,0,1,1,0,1,1,1,0,1,0,0,1,1,1,0,1,1,0,1,1,1],
#  [1,0,1,1,1,1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,0,1,0,1,1,1,0,1],
#  [1,0,1,1,0,0,0,0,1,1,1,1,0,0,1,0,1,1,1,0,1,1,0,0,0,1,1,1],
#  [1,1,0,1,1,0,1,0,1,1,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,1,0,1],
#  [1,0,1,1,1,1,0,1,1,1,0,1,1,0,1,1,1,1,0,0,1,1,1,1,0,1,0,1],
#  [0,1,1,1,0,1,1,1,0,1,1,0,1,0,1,1,0,0,0,0,1,1,1,0,1,1,1,1],
#  [1,1,1,1,0,0,1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,1,1,0,1,1,1,1],
#  [1,1,1,1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1,0],
#  [1,0,1,1,1,1,0,0,1,1,0,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,0,1],
#  [1,1,0,1,1,1,1,0,1,1,0,0,0,1,1,0,1,0,1,1,0,1,1,1,1,0,1,1],
#  [0,1,0,1,0,1,1,1,1,0,1,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,1,1],
#  [1,1,1,1,0,1,1,1,1,0,1,1,1,0,0,1,0,1,0,1,1,1,1,0,1,1,1,0],
#  [0,0,1,1,1,1,0,0,1,1,1,0,0,1,1,1,1,0,1,0,1,1,1,1,0,1,1,0],
#  [0,1,1,0,1,0,1,1,1,1,0,1,0,1,0,1,1,0,1,1,0,1,0,1,0,1,1,1],
#  [1,1,1,1,0,1,0,1,1,1,1,0,0,1,0,1,1,1,0,1,0,1,1,1,0,0,1,1],
#  [0,1,1,0,1,0,1,0,1,1,1,0,0,1,1,0,0,1,1,0,1,0,1,1,1,0,1,1],
#  [1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1],
#  [0,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,0,0,1,1,1,1,0,1,0,1,1,1],
#  [1,0,1,0,1,0,1,0,1,1,1,0,1,1,0,0,1,1,1,0,0,1,1,0,1,1,1,1],
#  [1,0,1,1,1,1,0,0,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,1,0],
#  [0,1,0,1,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1,0,1],
#  [1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1],
#  [1,1,1,1,0,0,1,0,1,0,1,1,1,1,0,1,0,1,0,1,1,1,1,0,1,1,1,0],
#  [0,0,1,1,1,0,0,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,0,0,1,1],
#  [1,1,1,1,0,0,0,1,1,0,1,1,0,1,1,0,1,0,1,1,1,0,1,1,1,1,0,1],
#  [1,1,0,1,0,0,1,1,1,1,0,1,1,0,1,1,0,1,1,1,1,0,1,1,0,1,0,1],
#  [0,1,1,1,1,0,1,1,1,0,1,1,0,1,1,1,0,1,0,1,1,1,1,0,1,1,1,1],
#  [0,1,1,0,1,0,1,1,1,1,0,1,1,0,0,1,0,1,1,1,1,0,1,0,1,1,1,1],
#  [1,1,1,0,0,1,1,1,0,1,0,1,1,1,1,0,1,1,1,0,1,1,0,1,1,0,1,1],
#  [1,1,0,1,0,1,0,1,1,1,1,0,0,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1],
#  [0,1,0,1,1,1,0,1,1,0,1,1,0,1,1,1,0,1,1,1,0,0,1,1,0,1,0,0],
#  [0,1,1,1,1,0,1,0,1,1,1,0,1,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1],
#  [1,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,1,0,0,0,1,1,0,1,1,0],
#  [1,0,0,1,1,1,1,0,1,1,1,0,1,1,1,1,0,0,0,1,0,1,0,1,1,1,0,1],
#  [1,1,0,1,0,0,1,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,1,0,0,1,1],
#  [0,1,1,1,1,0,1,0,1,1,0,1,1,1,1,0,1,0,1,1,0,1,1,1,1,0,1,1],
#  [1,1,1,1,0,1,1,1,0,1,1,1,1,0,1,0,1,1,1,0,1,1,1,1,0,1,1,1],
#  [1,1,1,0,1,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,0,1,0,1,1,1,1],
#  [1,1,1,0,1,1,1,1,0,1,1,1,1,0,0,1,1,0,1,1,0,1,1,1,1,0,1,0],
#  [1,1,0,1,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,1,1,1,0,1,1,1,0,1],
#  [1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,1,1,0,1,1,1,1,0,1,1,1,1,0],
#  [1,0,1,1,1,1,0,1,1,0,1,1,1,1,0,1,1,1,0,1,1,0,1,1,1,0,1,1],
#  [1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,0,1,1,1,0,1,1,1],
#  [0,1,1,1,0,1,1,1,1,0,1,1,0,1,1,0,1,1,1,1,0,0,1,1,1,1,0,1],
#  [1,1,0,0,1,0,1,1,1,0,1,0,1,1,1,1,0,1,0,1,1,1,0,1,1,0,1,1],
#  [1,1,0,1,0,1,1,1,1,0,1,1,0,0,0,1,1,0,1,1,1,1,0,1,1,0,1,1],
#  [1,1,1,0,1,0,1,1,1,1,0,1,0,1,1,1,1,0,1,0,1,0,1,1,0,1,1,1],
#  [1,1,1,0,1,1,1,0,1,1,1,0,1,1,0,1,1,1,1,0,0,1,1,1,0,1,1,1],
#  [1,1,1,0,0,0,1,1,1,1,0,1,1,1,1,0,1,0,1,0,1,1,1,1,0,0,1,1],
#  [1,1,1,0,1,0,0,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1,0,0,1,1,1,1],
#  [1,0,1,1,1,0,1,1,1,1,0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,1,1],
#  [1,1,0,1,1,0,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,0,1,0],
#  [1,0,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,1,0]]


# if __name__ == '__main__':
#     import pandas as pd
#     import sys
#     filename = 'best_solution1663438454.6025019'
#     df = pd.read_csv(filename, sep=' ')
#     df = df.round(4).astype(float).astype(int)
#     np.set_printoptions(threshold=sys.maxsize)
#     print(np.array2string(df.values, separator=","))
