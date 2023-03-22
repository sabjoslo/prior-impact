This repository contains all the code necessary to reproduce the figures in Sloman, S. J., Cavagnaro, D., and Broomell, S. B. (2023). *Knowing what to know: Implications of the choice of prior distribution on the behavior of adaptive design optimization* [Unpublished manuscript].

# Dependencies

This code relies on the [pyBAD](https://github.com/sabjoslo/pyBAD) package and its dependencies.

# Reproducing figures

The folders `irt` and `memreten` contain all the code necessary to replicate the results from simulation experiments using the item-reponse and memory retention modeling paradigms, respectively. The Python script `runSimulations.py` provides the necessary infrastructure to run the simulation experiments. The Python script `buildDataMatrix.py` processes the output of these simulation files. The Jupyter notebook `results.ipynb` displays the results.

The Jupyter notebook `figure6.ipynb` reproduces the toy example shown in Figure 6.