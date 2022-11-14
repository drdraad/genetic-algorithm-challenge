# optimisation experiment code
import random
import time
from opt_params_EDIT import *
from genetic_algorithm_EDIT import GA


def run_experiment(experiment_name: str,
                   num_runs=num_independent_runs,
                   run_timeout=run_timeout_seconds,
                   random_seed=1,
                   write_best_solution=False,
                   ):
    """
    Run num_runs optimisation experiments with a run_timeout seconds budget
    :param experiment_name: the name of the experiment that will be output
    :param num_runs: number of independent trials with different random starting conditions
    :param run_timeout: time budget per run in seconds
    :param random_seed: random seed to initialise random number generators
    :param write_best_solution: if True then write the best solution to txt file
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    ga = GA(minimisation_objective_func=calculate_cookie_wages,
            named_constraint_funcs=named_constraint_funcs,
            constraint_penalty_weights=penalty_weights,
            pop_size=pop_size,
            elitism_size=elitism_size,
            mutation_rate=mutation_rate,
            decision_var_shape=(dim1, dim2),
            random_init_func=random_shifts_initialisation,
            include_design_knowledge=True
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
        best_fitness_history = [(1, best_fitness)]
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
            best_fitness_history.append((round(time.time() - start_time), pop_fitness[0]))
            generation += 1
            generations.append(generation)

            print("Run: " + str(run+1) + " | Generation: " + str(generation) + " | Best fitness: " + str(
                np.round(best_fitness_history[-1][1], 2)))

        best_solutions.append(best_solution)
        best_fitnesses.append(best_fitness_history[-1][1])
        best_fitness_only = [b[1] for b in best_fitness_history]
        time_to_minimum.append(np.where(best_fitness_only == best_fitness_only[-1])[0][0]
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

    # calculate average fitness per second
    avg_best_fitness_per_second = np.zeros(run_timeout)
    for i in range(run_timeout):
        for h in best_fitness_histories:
            avg_fitness_at_second_i = np.mean([f[1] for f in h if f[0] == i+1])
            avg_best_fitness_per_second[i] += avg_fitness_at_second_i/num_runs

    print(avg_best_fitness_per_second)

    if write_best_solution:
        np.save(experiment_name + '_best_solution'+str(time.time()),
                best_solution)
        np.save(experiment_name + '_avg_best_fitness_per_second'+str(time.time()),
                avg_best_fitness_per_second)

    return np.mean(best_fitnesses)


if __name__ == "__main__":
    run_experiment(experiment_name='base_design')
