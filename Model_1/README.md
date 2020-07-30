# A contextual bandit model for network slicing - Model 1 

This folder contains the simulation program of a contextual bandit model for 
network slicing. 

## Content of files

main.py contains the simulation setup. Model parameters can be modified in this file.

Game.py implements the main body of a contextual bandit model for network slicing. 

Particle_Thompson_Sampling.py implements the particle Thompson sampling algorithm.

auxuliary.py contains some auxiliary functions. 

## How to use

1. Open main.py and set the parameters: 

D: a positive integer, number of domains
B: a positive integer list, the number of resource blocks in each domain
t: the time horizon (number of steps in each simulation)
N_simul: the number of simulations to run, over which the average regret will be obtained
