# This code is part of LINKEQ.
#
# (C) Copyright LINKE 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# -*- coding: utf-8 -*-
# @Time     : 2024/1/22 11:22
# @Author   : HFLASH @ LINKE
# @File     : tut_cd.py
# @Software : PyCharm


import main_cd as main

# Other modules
import numpy as np
import numpy.matlib as npm
import json
import matplotlib.pyplot as plt
from matplotlib import rc
import copy
import warnings
import scipy
from scipy import sparse
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdmn
import importlib as imp
import os
import random

# Save figures in the notebook with decent quality
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 100

# Create figures directory if needed
try:
    os.mkdir('figs')
except FileExistsError:
    pass

imp.reload(main)

## PROTOCOL
protocol = 'srs' # Currently only 'srs' has been debugged

## TOPOLOGY
# Use any function main.adjacency_*() to define a topology.
# Here we use a squared lattice (with hard boundary conditions)
# with 9 nodes as an example.
l = 3
n = int(l*l)
A = main.adjacency_squared_hard(l)
topology = 'squared_hard'

## HARDWARE
p_gen = 0.5 # Probability of successful entanglement generation
p_swap = 1 # Probability of successful swap
qbits_per_channel = 5 # Number of qubits per node per physical neighbor

## SOFTWARE
q_swap = 0.12 # Probability of performing swaps in the SRS protocol
max_links_swapped = 4 # Maximum number of elementary links swapped
p_cons = 0.1 # Probability of virtual neighbors consuming a link per time step
F_app = 0.6 # Minimum fidelity required by the background application

## CUTOFF
# The cutoff is here chosen arbitrarily. To find a physically meaningful value,
# one should use the coherence time of the qubits and the fidelity of newly
# generated entangled links. A valid approach is to use a worst-case model
# as in Iñesta et al. 'Optimal entanglement distribution policies in homogeneous
# repeater chains with cutoffs', 2023.
cutoff = 20

## SIMULATION
data_type = 'avg' # Store only average (and std) values instead of all simulation data
N_samples = 100 # Number of samples
total_time = int(cutoff*5) # Simulation time
plot_nodes = [0,4,5] # We will plot the time evolution of these nodes
randomseed = 2
np.random.seed(randomseed)