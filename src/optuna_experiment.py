# optimisation experiment code
import random
import time
from opt_params_improved import *
from genetic_algorithm_improved import ImprovedGA
import optuna


def hpo_experiment(trial,
                   num_runs=num_independent_runs,
                   run_timeout=run_timeout_seconds,
                   random_seed=1,
                   output_best_solution=False,
                   ):
    """
    Conduct HPO on Genetic Algorithm params using optuna library (Tree-structured Parzen Estimator)
    Run num_runs optimisation experiments with a run_timeout seconds budget
    :param trial: optuna sampling object per trial
    :param num_runs: number of independent trials with different random starting conditions
    :param run_timeout: time budget per run in seconds
    :param random_seed: random seed to initialise random number generators
    :param output_best_solution: if True then write the best solution to txt file
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    o_include_designs = False #trial.suggest_categorical("designs", [True, False])
    o_repair_prob = trial.suggest_float("repair_prob", 0.003, 0.004)
    o_random_injection_prob = trial.suggest_float("random_injection_probability", 0.008, 0.015)
    o_pop_size = trial.suggest_int("pop_size", 30, 34)
    o_elite_size = trial.suggest_int("elite_size", 29, o_pop_size)
    o_mut_rate = trial.suggest_float("mut_rate", 0.001, 0.002)
    o_swap_prob = trial.suggest_float("swap_prob", 0.1, 0.2)
    o_num_swaps = trial.suggest_int("num_swaps", 6, 12)

    ga = ImprovedGA(minimisation_objective_func=calculate_cookie_wages,
                    named_constraint_funcs=named_constraint_funcs,
                    constraint_penalty_weights=penalty_weights,
                    pop_size=o_pop_size,
                    elitism_size=o_elite_size,
                    mutation_rate=o_mut_rate,
                    decision_var_shape=(dim1, dim2),
                    random_init_func=random_shifts_initialisation,
                    random_injection_probability=o_random_injection_prob,
                    repair_probability=o_repair_prob,
                    include_design_knowledge=o_include_designs,
                    swap_prob=o_swap_prob,
                    num_swaps=o_num_swaps,
                    )

    best_solutions = []
    best_fitnesses = []
    best_fitness_histories = []
    time_to_minimum = []

    # outer loop for independent runs
    for run in range(num_runs):

        start_time = time.time()
        # randomly initialise population
        population = ga.random_initialisation()
        pop_fitness = ga.evaluate_fitness(population)
        best_fitness = min(pop_fitness)
        best_fitness_history = [best_fitness]
        idx = np.where(pop_fitness == best_fitness)[0][0]
        best_solution = population[:, :, idx]
        generation = 1
        generations = [1]

        # inner loop for GA generations
        while time.time() < start_time + run_timeout:
            # randomly generate child solutions from parents via crossover and mutation
            children, children_fitness = ga.generate_children(population, pop_fitness)

            # select solutions to survive into the next generation
            population, pop_fitness = ga.survival(population, pop_fitness, children, children_fitness)

            # store best found solution and fitness (always at pop index 0)
            best_solution = population[:, :, 0]
            best_fitness_history.append(pop_fitness[0])
            generation += 1
            generations.append(generation)

            # print("Run: " + str(run+1) + " | Generation: " + str(generation) + " | Best fitness: " + str(
            #     np.round(best_fitness_history[-1], 2)))

        best_solutions.append(best_solution)
        best_fitnesses.append(best_fitness_history[-1])
        time_to_minimum.append(np.where(best_fitness_history == best_fitness_history[-1])[0][0]
                               / len(best_fitness_history) * run_timeout)
        best_fitness_histories.append(best_fitness_history)

    # experiment performance summary
    best_solution_idx = np.where(best_fitnesses == min(best_fitnesses))[0][0]
    best_solution = best_solutions[best_solution_idx]
    print("EXPERIMENT RESULTS")
    print("Average fitness over " + str(num_runs) + " " + str(run_timeout) +
          "-second runs: " + str(np.round(np.mean(best_fitnesses), 2)))
    print("Variance of fitness over " + str(num_runs) + " " + str(run_timeout) +
          "-second runs: " + str(np.var(best_fitnesses)))
    print("Average time to minimum: " + str(np.round(np.mean(time_to_minimum), 2)))
    print("Number of disappointed children this Xmas: " + str(ga.named_constraints["Toy shortfall"](best_solution)))
    print("BEST SOLUTION REPORT")
    ga.solution_performance_report(best_solution)
    return np.mean(best_fitnesses)


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(hpo_experiment, n_trials=100)
    print(study.best_trial)
