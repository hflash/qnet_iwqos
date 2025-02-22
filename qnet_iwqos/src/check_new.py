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
# @Time     : 2025/1/9 15:00
# @Author   : HFLASH @ LINKE
# @File     : check_new.py
# @Software : PyCharm

def simulation_cd_for_virtual_neighbors(protocol, A, p_gen, q_swap, p_swap, p_cons, cutoff, M, qbits_per_channel,
                                        N_samples, total_time,
                                        randomseed, progress_bar=None, return_data='all'):
    ''' ---Inputs---
            · protocol: (str) protocol to be run ('srs' or 'ndsrs' or cfs).
            · A:    (array) physical adjacency matrix.
            · p_gen: (float) probability of successful entanglement generation.
            · q_swap:   (float) probability that a swap is attempted. In the 'ndsrs',
                        this has to be a list of length len(A).
            · p_swap:   (float) probability of successful swap.
            · p_cons:   (float) probability of link consumption.
            · cutoff:   (int) cutoff time.
            · M:    (int) maximum swap length.
            · qbits_per_channel:    (int) number of qubits per node reserved
                                    for each physical channel.
            · N_samples:    (int) number of samples.
            · total_time:
            · progress_bar: (str) None, 'notebook', or 'terminal'.
            · return_data:  (str) 'avg' or 'all'.
        ---Outputs---
            '''
    n = len(A)
    if progress_bar == None:
        _tqdm = lambda *x, **kwargs: x[0]
    elif progress_bar == 'notebook':
        _tqdm = tqdmn
    elif progress_bar == 'terminal':
        _tqdm = tqdm
    else:
        raise ValueError('Invalid progress_bar')

    # np.random.seed(randomseed)
    # Calculate physical degrees
    pdegrees = physical_degrees(A)

    if return_data == 'all':
        # Initialize time-dependent node-dependent virtual quantities
        vdegrees = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
        vneighs = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
        vneighborhoods = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]

        # ISSUE: If there is an error, may need to uncomment the four lines below
        # avg_vdegrees = [[0 for _ in range(total_time+1)] for _ in range(n)]
        # avg_vneighs = [[0 for _ in range(total_time+1)] for _ in range(n)]
        # std_vdegrees = [[0 for _ in range(total_time+1)] for _ in range(n)]
        # std_vneighs = [[0 for _ in range(total_time+1)] for _ in range(n)]
        for sample in _tqdm(range(N_samples), 'Samples', leave=False):

            S = create_qubit_registers(A, qbits_per_channel)
            for t in range(0, total_time):
                if protocol == 'srs':
                    S, count = step_protocol_srs(S, p_gen, q_swap, p_swap, p_cons, cutoff, M, randomseed)
                elif protocol == 'ndsrs':
                    S = step_protocol_ndsrs(S, p_gen, q_swap, p_swap, p_cons, cutoff, M)
                else:
                    raise ValueError('Protocol not implemented')
                for node in range(n):
                    vdeg, vneigh, vneighborhood, vneigh_links = virtual_properties(S, node)
                    vdegrees[node][t][sample] = vdeg
                    vneighs[node][t][sample] = vneigh
                    vneighborhoods[node][t][sample] = vneighborhood
        # avg_vdegrees = np.mean(vdegrees, axis=2)
        # avg_vneighs = np.mean(vneighs, axis=2)
        # std_vdegrees
        # = np.std(vdegrees, axis=2)
        # std_vneighs = np.std(vneighs, axis=2)

        return vdegrees, vneighs, vneighborhoods, None