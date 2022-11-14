from problem import *

# EDIT TO YOUR HEART'S CONTENT vvv ############################################

pop_size = 30
elitism_size = 30
mutation_rate = 0.0012450584918284457
random_injection_probability = 0.012932496150143327
repair_prob = 0.0037938633052786923
swap_prob = 0.18426302343522813
num_swaps = 9
incl_designs = False

# EDIT TO YOUR HEART'S CONTENT ABOVE ^^^ ######################################


# DO NOT EDIT BELOW vvv #######################################################

# experiment params
num_independent_runs = 5
run_timeout_seconds = 60

# suggested constraint penalty weights
# (typically you might modify these, but we're keeping it simple for the challenge - see notebook for analysis)
toy_shortfall_penalty_weight = 73956.56
workshop_capacity_violation_weight = 73956.56
consecutive_shift_violation_weight = 73956.56
elf_max_min_ratio_violation_weight = 1320636.0

# decision variables dimensions
dim1 = num_junior_elves + num_senior_elves
dim2 = num_shifts

# data structures
penalty_weights = [toy_shortfall_penalty_weight,
                   workshop_capacity_violation_weight,
                   consecutive_shift_violation_weight,
                   elf_max_min_ratio_violation_weight]

named_constraint_funcs = {"Toy shortfall": calculate_toy_shortfall,
                          "Workshop capacity": workshop_capacity_violation,
                          "Consecutive shifts": consecutive_shift_violation,
                          "Max/min work ratios": elf_max_min_ratio_violation,
                          }

# DO NOT EDIT ABOVE ^^^ #######################################################

# if __name__ == '__main__':
#
#     import time
#     import random
#
#     t0 = time.time()
#
#     for i in range(10000):
#         rand_matrix = np.random.rand(200, 28)
#
#     t1 = time.time()
#
#     for i in range(10000):
#         rand_matrix = random.random(200, 28)
#
#     t2 = time.time()
#
#     print((t2-t1)/(t1-t0))
