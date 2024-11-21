'''Run CD protocol simulation and save data.
   Alvaro G. Inesta. TU Delft, 2022.'''

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
import pandas as pd
from sys import getsizeof
import random
import argparse

import main_cd as main

#------------------------------------------------------------------------------
# ARGPARSER
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--protocol', type=str, default='srs',
                    help='Protocol (e.g., srs).')

# Topology
parser.add_argument('--topology', type=str, default='tree',
                    help='Physical topology.')
parser.add_argument('--n', type=int, default=23,
                    help='Number of nodes. If topology is a (d,k)-tree,'\
                        'd is the first digit in FLAGS.n and k is the second,'\
                        'digit. If waxman, the string should have the form'\
                        '"waxman"+str(max_dist)+boundary+str(seed)+"beta"+'\
                        'str(beta)+"alphaL"+str(alphaL)')

# Hardware
parser.add_argument('--p_gen', type=float, default=0.2,
                    help='Probability of successful '\
                            'heralded entanglement generation.')
parser.add_argument('--p_swap', type=float, default=1,
                    help='Probability of successful swap.')

# Software
parser.add_argument('--q_swap', type=float, default=0.1,
                    help='Probability of performing each swap.')
parser.add_argument('--p_cons', type=float, default=0.1,
                    help='Probability of consuming a link '\
                            'between each pair of virtual neighbors.')
parser.add_argument('--cutoff', type=int, default=221,
                    help='Cutoff time.')
parser.add_argument('--max_links_swapped', type=int, default=3,
                    help='Maximum number of elementary links swapped '\
                            'to form a single longer link (also called M'\
                            'in our equations).')
parser.add_argument('--qbits_per_channel', type=int, default=5,
                    help='Number of qubits per node allocated to each '\
                            'physical channel (also called r'\
                            'in our equations).')

# Numerical
parser.add_argument('--N_samples', type=int, default=100,
                    help='Number of samples.')
parser.add_argument('--total_time', type=int, default=442,
                    help='Total simulation time in time steps.')
parser.add_argument('--randomseed', type=int, default=2,
                    help='Random seed.')
parser.add_argument('--data_type', type=str, default='avg',
                    help='If "all", saves all data.'\
                            'If "avg", saves averages and stds over runs.')

f = parser.parse_args()

#------------------------------------------------------------------------------
# CALCULATIONS
#------------------------------------------------------------------------------
np.random.seed(f.randomseed)
random.seed(f.randomseed)

#print('LOOPING OVER p_cons')
#for p_cons in np.arange(0,0.301,0.025):
#    print('p_cons = %.3f'%p_cons)
#    f.p_cons = p_cons

# Check if data exists
if not main.check_data_cd(f.protocol, f.data_type, f.topology, f.n, f.p_gen,
            f.q_swap, f.p_swap, f.p_cons, f.cutoff, f.max_links_swapped,
            f.qbits_per_channel, f.N_samples, f.total_time, f.randomseed):
    
    # Find adjacency matrix
    if f.topology == 'chain':
        A = main.adjacency_chain(f.n)
    elif f.topology == 'squared':
        A = main.adjacency_squared(np.sqrt(f.n))
    elif f.topology == 'tree':
        d = int(str(f.n)[0])
        k = int(str(f.n)[1])
        A = main.adjacency_tree(d,k)
    elif f.topology[0:6] == 'waxman':
        if f.topology.find('circle') == -1:
            raise ValueError('Unknown boundary')
        else:
            boundary = 'circle'
        max_dist = int(f.topology[len('waxman'):f.topology.find('circle')])
        seed = int(f.topology[f.topology.find('circle')+len('circle'):f.topology.find('beta')])
        beta = float(f.topology[f.topology.find('beta')+len('beta'):f.topology.find('alphaL')])
        alphaL = int(f.topology[f.topology.find('alphaL')+len('alphaL'):])
        A, _ = main.adjacency_waxman(f.n, max_dist, boundary, seed, beta, alphaL)
    else:
        raise ValueError('Unknown topology')

    # Compute
    data = main.simulation_cd(f.protocol, A, f.p_gen, f.q_swap, f.p_swap,
                                f.p_cons, f.cutoff, f.max_links_swapped,
                                f.qbits_per_channel, f.N_samples,
                                f.total_time,
                                progress_bar='terminal',
                                return_data=f.data_type)
    # Save data
    main.save_data_cd(data, f.protocol, f.data_type, f.topology, f.n,
                    f.p_gen, f.q_swap, f.p_swap, f.p_cons, f.cutoff,
                    f.max_links_swapped, f.qbits_per_channel, f.N_samples,
                    f.total_time, f.randomseed)
    print('Done! Data saved!')
else:
    print('Data already exists!')























