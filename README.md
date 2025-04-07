## Genetic Algorithm Santa's Elf Scheduling Challenge

### Overview

Welcome to the GA Elf Scheduling Challenge! The genetic algorithm tutorial and challenge details appear in [genetic_algorithm_challenge.ipynb](https://github.com/drdraad/genetic-algorithm-challenge/blob/solution/genetic_algorithm_challenge.ipynb)

### Getting Started

1. Run `pip install -r requirements.txt` to install required packages.

2. The code defining the problem is in `src/problem.py` and the algorithm testing code is in `opt_experiment.py`

3. You are only required to edit the files `src/genetic_algorithm_EDIT.py` and `src/opt_params_EDIT.py` for this challenge in the sections marked _# TODO: ADD CODE HERE_. 

4. You may optionally also overwrite the `generate_children()` function inherited from `src.base.BaseGA`. 

5. You may also write your own script to conduct hyperparameter optimisation on the GA's parameters.

### Solution
Solution on `solution` branch (don't peek!), including `ga-challenge-2022-debrief.pdf` showing the performance of different GA variants over time.
 
The following files should be run when testing:

- `opt_experiment_base.py` - Run an optimisation experiment for the base GA
- `opt_experiment_improved.py` - Run an optimisation experiment for the improved GA
- `optuna_experiment.py` - Conduct HPO on the parameters of the improved GA
- `utils.py` - Generate a graph comparing the average performance of the different algorithms over time

Note: _Benchmarking was conducted on a MacBook Pro, macOS Monterey 12.5.1 | 2 GHz Quad-Core Intel Core i5 | 16 GB 3733 MHz LPDDR4X_
