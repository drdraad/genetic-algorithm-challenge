## Genetic Algorithm Challenge 2022

### Overview

Genetic algorithm tutorial and challenge details appear in `genetic_algorithm_challenge.ipynb`

### Getting Started

Run `pip install -r requirements.txt` to install required packages.

The code defining the problem to solve is in `src/problem.py` and the algorithm testing code is in `opt_experiment.py`

You are only required to edit the files `src/genetic_algorithm_EDIT.py` and `src/opt_params_EDIT.py` for this challenge in the sections marked _# TODO: ADD CODE HERE_.

### Solution
Solution on `solution` branch (don't peek!), including `ga-challenge-2022-debrief.pdf` analysing the performance of the algorithms over time.
 
The following files should be run when testing:

- `opt_experiment_base.py` - Run an optimisation experiment for the base GA
- `opt_experiment_improved.py` - Run an optimisation experiment for the improved GA
- `optuna_experiment.py` - Conduct HPO on the parameters of the improved GA
- `utils.py` - Generate a graph comparing the average performance of the different algorithms over time

Note: Benchmarking was conducted on a MacBook Pro, macOS Monterey 12.5.1 | 2 GHz Quad-Core Intel Core i5 | 16 GB 3733 MHz LPDDR4X
