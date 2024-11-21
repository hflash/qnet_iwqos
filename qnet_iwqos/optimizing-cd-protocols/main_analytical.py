'''Main analytical functions to analyze CD protocols.
   Alvaro Gomez Inesta. TU Delft, 2022.'''

import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdmn


def steady_state_vector(pr_fwd_array, pr_bck_array):
    '''Steady state vector calculation in a general random walk.
        Element w is the probability of being in state w
        in the steady state.
        ---Inputs---
            · pr_fwd_array: (float list) forward transition probabilities.
            · pr_bck_array: (float list) backward transition probabilities.'''
    assert pr_bck_array[0] == 0, 'pr_bck_array[0] should be 0'
    assert pr_fwd_array[-1] == 0, 'pr_fwd_array[-1] should be 0'
    assert len(pr_fwd_array) == len(pr_bck_array)

    # Avoid singularities by replacing zeroes by tiny numbers
    for idx, p in enumerate(pr_fwd_array):
        if p==0:
            pr_fwd_array[idx] = 1e-15
    for idx, q in enumerate(pr_bck_array):
        if q==0:
            pr_bck_array[idx] = 1e-15

    gamma = len(pr_fwd_array)-1
    ss_vector = np.zeros(gamma+1)

    # State 0
    denom = 1 + np.sum( [np.prod([ pr_fwd_array[m]/pr_bck_array[m+1] \
                                    for m in range(0,k) ]) \
                            for k in range(1,gamma+1)] )
    pi_0 = 1 / denom
    ss_vector[0] = pi_0

    # States 1 to gamma
    for state in range(1,gamma+1):
        numer = np.prod( [pr_fwd_array[m]/pr_bck_array[m+1] \
                            for m in range(0,state)] )
        ss_vector[state] = numer * pi_0

    return ss_vector


#-------------------------------------------------------
#--------------------- NO SWAPS ------------------------
#-------------------------------------------------------

def steady_state_noswaps(p_gen, p_cons, gamma):
    '''Steady state vector with no swaps, infinite cutoff,
        and gamma=min(cutoff,r).'''
    gamma = int(gamma)
    ss_vector = np.zeros(gamma+1)

    if p_gen==1: # To avoid singularities
        p_gen-=1e-10

    # State 0
    ss_vector[0] = (p_gen-p_cons) / ((1-p_gen) * (p_gen*
                    (p_gen*(1-p_cons)/(p_cons*(1-p_gen)))**gamma - p_cons))

    rho = p_gen*(1-p_cons) / (p_cons*(1-p_gen))

    # States 1 to gamma-1
    for state in range(1,gamma):
        ss_vector[state] = ss_vector[0] * rho**state

    # State gamma
    ss_vector[-1] = ss_vector[0] * rho**gamma * (1-p_gen)

    return ss_vector

def virtual_ss_noswaps(physical_degree, p_gen, p_cons,
                        cutoff, qbits_per_channel):
    '''---Inputs---
        · physical_degree:  (int) physical degree of a node.'''
    with np.errstate(divide='ignore', invalid='ignore'):
    #if True:
        if cutoff < 10*1/p_cons and p_cons!=0:
            print('WARNING: cutoff (%s) must be much larger '\
                    'than 1/p_cons (%.1f)'%(cutoff,1/p_cons))

        gamma = min(cutoff, qbits_per_channel)
        
        vneighs = physical_degree * (p_gen**(gamma+1) * (1-p_cons)**gamma
                    - p_gen*(1-p_gen)**(gamma-1)*p_cons**gamma
                    *(1-p_cons) ) / (p_gen**(gamma+1)*(1-p_cons)**gamma
                    - (1-p_gen)**gamma*p_cons**(gamma+1) )

        vdeg = physical_degree * p_gen * ( gamma * (p_gen-p_cons)*p_gen**gamma
                *(1-p_cons)**gamma + p_cons*(1-p_cons)*( p_cons**gamma
                    *(1-p_gen)**gamma - p_gen**gamma
                    *(1-p_cons)**gamma ) ) / ( (p_gen-p_cons)
                *(p_gen**(gamma+1)*(1-p_cons)**gamma
                    - p_cons**(gamma+1)*(1-p_gen)**gamma) )

        ss_vector = steady_state_noswaps(p_gen, p_cons, gamma)
        if not (np.isnan(ss_vector)).any(): # If p_cons=0, ss_vector is nan
            if np.abs(vneighs - physical_degree * (1-ss_vector[0])) > 1e-3:
                print('.---vneigh---')
                print(p_cons)
                print(vneighs)
                print(physical_degree * (1-ss_vector[0]))
            assert np.abs(vneighs - physical_degree * (1-ss_vector[0])) < 1e-3
            if np.abs(vdeg - physical_degree * sum(
                    np.multiply(ss_vector,np.arange(0,
                        len(ss_vector),1)))) > 1e-2:
                print('.---vdeg---')
                print(p_cons)
                print(vdeg)
                print(physical_degree * sum(
                    np.multiply(ss_vector,np.arange(0,
                        len(ss_vector),1))))
            assert np.abs(vdeg - physical_degree * sum(
                    np.multiply(ss_vector,np.arange(0,
                        len(ss_vector),1)))) < 1e-2

    return vneighs, vdeg

#-------------------------------------------------------
#-------------------------------------------------------
#-------------------------------------------------------

def leading_order_trans_prob_srs(p_cons, q_swap, gamma, d_i, d_j):
    '''Leading-order estimation of transition probabilities
        for the general random walk modeling the number of links
        between nodes i and j, in a SRS protocol.
        ---Inputs---
            · p_cons:   (float) consumption probability.
            · q_swap:   (float) probability of performing a swap.
            · gamma:    (int) maximum number of memories per node:
                        gamma=min(cutoff,r).
            · d_i:  (int) physical node degree of node i.
            · d_j:  (int) physical node degree of node j.'''
    pr_fwd_array = list(np.zeros(gamma+1))
    pr_bck_array = list(np.zeros(gamma+1))

    # State 0
    pr_fwd_array[0] = (1-p_cons) * (1-2*q_swap/d_j) * (1-2*q_swap/d_i)

    # States 1 to gamma-1
    for state in range(1,gamma):
        pr_fwd_array[state] = (1-p_cons) * (1-2*q_swap/d_j) * (1-2*q_swap/d_i)
        pr_bck_array[state] = 2 * p_cons * q_swap * (1/d_j + 1/d_i) \
                                + 4 * q_swap**2 / (d_i*d_j)

    # State gamma
    pr_bck_array[-1] = p_cons * ( 1 - 4*q_swap*(1/d_j + 1/d_i) ) \
                        + 2 * q_swap * (1/d_i + 1/d_j) \
                        - 4 * q_swap**2 / (d_i*d_j)

    return pr_fwd_array, pr_bck_array

def cross_entropy(p1, p2):
    '''Computes the cross-entropy between the probability distributions
        p1 and p2. When both distributions are the same, the cross-entropy
        is the entropy of the distribution.'''
    assert len(p1)==len(p2)

    return -np.sum([p1[idx]*np.log(p2[idx]) for idx in range(len(p1))])

def KL_divergence(p1, p2):
    '''Computes the KL divergence between the probability distributions
        p1 and p2. When both distributions are the same, the KL divergence
        is zero.'''
    assert len(p1)==len(p2)

    KL_div = 0
    for idx in range(len(p1)):
        if p1[idx]==0:
            pass
        else:
            KL_div -= p1[idx]*np.log(p2[idx]/p1[idx])

    return KL_div
























