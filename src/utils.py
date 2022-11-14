import matplotlib.pyplot as plt
import numpy as np


def plot_fitness(avg_fitness_per_second: dict, num_secs: int=60, save_fig=False):
    """
    Plots the average fitness per second per algorithm
    :param avg_fitness_per_second: dict of the form {"run-name": np.array([])}
    :param num_secs: the number of the seconds of the run
    """
    x = np.arange(1, num_secs+1)
    for key, value in avg_fitness_per_second.items():
        plt.plot(x, value, label=key)
        # plt.fill_between(x, value - stddev, value + stddev)
    plt.yscale('log')
    plt.ylabel('Log fitness value')
    plt.xlabel('Time in seconds')
    plt.title('Genetic Algorithm Average Best Fitness per Second')
    plt.legend()
    if save_fig:
        plt.savefig(str(len(avg_fitness_per_second)) + '.png')
    plt.show()


def load_experiment_files(filenames: list):
    avg_fitness_per_second = {}
    for f in filenames:
        start = f.find('t/') + 2
        end = f.find('_avg')
        avg_fitness_per_second[f[start:end]] = np.load(f)
    print(avg_fitness_per_second.keys())
    plot_fitness(avg_fitness_per_second)


if __name__ == '__main__':
    files = [
        '../output/untuned_avg_best_fitness_per_second1664226794.6873732.txt.npy',
        '../output/base_avg_best_fitness_per_second1664223529.586835.npy',
        '../output/base_numba_avg_best_fitness_per_second1664226032.0887098.txt.npy',
        '../output/base_design_avg_best_fitness_per_second1664228554.748769.txt.npy',
        '../output/improved_avg_best_fitness_per_second1664229710.7026129.npy',
        '../output/improved_plus_avg_best_fitness_per_second1664262393.331588.npy'
    ]
    load_experiment_files(files)
