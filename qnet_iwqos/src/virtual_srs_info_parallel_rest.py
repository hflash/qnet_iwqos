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
# @Time     : 2024/2/20 18:56
# @Author   : HFLASH @ LINKE
# @File     : virtual_srs_info.py
# @Software : PyCharm
import json
import os
import random
import time

from scipy.sparse.csgraph import dijkstra
from circuit2graph import circuitPartition
import main_cd_circuit_execution as cd
import numpy as np
from distributed_operation_circuit import srs_config_squared_hard, srs_config_chain, srs_config_tree


# def srs_config_squared_hard(l, p_gen, q_swap, qubit_per_channel, cutoff, randomseed):
#     srs_configurations = {}
#
#     ## TOPOLOGY
#     # Use any function main.adjacency_*() to define a topology.
#     # Here we use a squared lattice (with hard boundary conditions)
#     # with 9 nodes as an example.
#     # l = 2
#     n = int(l * l)
#     A = cd.adjacency_squared_hard(l)
#     topology = 'squared_hard'
#     srs_configurations['adj'] = A
#
#     ## HARDWARE
#     p_gen = 1  # Probability of successful entanglement generation
#     p_swap = 1  # Probability of successful swap
#     qbits_per_channel = qubit_per_channel  # Number of qubits per node per physical neighbor
#     srs_configurations['p_gen'] = p_gen
#     srs_configurations['p_swap'] = p_swap
#     srs_configurations['qubits'] = qbits_per_channel
#
#     ## SOFTWARE
#     # q_swap = 0.12  # Probability of performing swaps in the SRS protocol
#     # max_links_swapped = 4  #  Maximum number of elementary links swapped
#     p_cons = 0  # Probability of virtual neighbors consuming a link per time step
#     srs_configurations['q_swap'] = q_swap
#     srs_configurations['max_swap'] = 4
#     srs_configurations['p_cons'] = p_cons
#
#     ## CUTOFF
#     # cutoff = 20
#     srs_configurations['cutoff'] = cutoff
#     if randomseed is not None:
#         srs_configurations['randomseed'] = randomseed
#     else:
#         srs_configurations['randomseed'] = random.seed()
#
#     return srs_configurations


def virtual_srs_info_vneighbors(srs_configurations, N_samples, total_time):
    protocol = 'srs'
    N_samples = N_samples
    total_time = total_time
    ## TOPOLOGY
    # Use any function main.adjacency_*() to define a topology.
    A = srs_configurations['adj']

    ## HARDWARE
    p_gen = srs_configurations['p_gen']  # Probability of successful entanglement generation
    p_swap = srs_configurations['p_swap']  # Probability of successful swap
    qbits_per_channel = srs_configurations['qubits']  # Number of qubits per node per physical neighbor

    ## SOFTWARE
    q_swap = srs_configurations['q_swap']  # Probability of performing swaps in the SRS protocol
    max_links_swapped = srs_configurations['max_swap']  #  Maximum number of elementary links swapped
    p_cons = srs_configurations['p_cons']  # Probability of virtual neighbors consuming a link per time step

    ## CUTOFF
    cutoff = srs_configurations['cutoff']
    np.random.seed(srs_configurations['randomseed'])

    vdegrees, vneighs, vneighborhoods, _ = cd.simulation_cd_for_virtual_neighbors(protocol, A, p_gen, q_swap, p_swap,
                                                                                  p_cons, cutoff, max_links_swapped,
                                                                                  qbits_per_channel, N_samples,
                                                                                  total_time,
                                                                                  srs_configurations['randomseed'],
                                                                                  progress_bar=None, return_data='all')
    return [vdegrees, vneighs, vneighborhoods]

def virtual_cfs_info_vneighbors(srs_configurations, N_samples, total_time, swap_mode):
    protocol = 'cfs'
    N_samples = N_samples
    total_time = total_time
    ## TOPOLOGY
    # Use any function main.adjacency_*() to define a topology.
    A = srs_configurations['adj']

    ## HARDWARE
    p_gen = srs_configurations['p_gen']  # Probability of successful entanglement generation
    p_swap = srs_configurations['p_swap']  # Probability of successful swap
    qbits_per_channel = srs_configurations['qubits']  # Number of qubits per node per physical neighbor

    ## SOFTWARE
    q_swap = srs_configurations['q_swap']  # Probability of performing swaps in the SRS protocol
    max_links_swapped = srs_configurations['max_swap']  #  Maximum number of elementary links swapped
    p_cons = srs_configurations['p_cons']  # Probability of virtual neighbors consuming a link per time step

    ## CUTOFF
    cutoff = srs_configurations['cutoff']
    np.random.seed(srs_configurations['randomseed'])

    vdegrees, vneighs, vneighborhoods, _ = cd.simulation_cd_for_virtual_neighbors_cfs(protocol, A, p_gen, q_swap, p_swap,
                                                                                  p_cons, cutoff, max_links_swapped,
                                                                                  qbits_per_channel, N_samples,
                                                                                  total_time, swap_mode,
                                                                                  srs_configurations['randomseed'],
                                                                                  progress_bar=None, return_data='all')

    return [vdegrees, vneighs, vneighborhoods]


def virtual_srs_info_avg(srs_configurations, N_samples, total_time):
    protocol = 'srs'
    N_samples = N_samples
    total_time = total_time
    ## TOPOLOGY
    # Use any function main.adjacency_*() to define a topology.
    A = srs_configurations['adj']

    ## HARDWARE
    p_gen = srs_configurations['p_gen']  # Probability of successful entanglement generation
    p_swap = srs_configurations['p_swap']  # Probability of successful swap
    qbits_per_channel = srs_configurations['qubits']  # Number of qubits per node per physical neighbor

    ## SOFTWARE
    q_swap = srs_configurations['q_swap']  # Probability of performing swaps in the SRS protocol
    max_links_swapped = srs_configurations['max_swap']  #  Maximum number of elementary links swapped
    p_cons = srs_configurations['p_cons']  # Probability of virtual neighbors consuming a link per time step

    ## CUTOFF
    cutoff = srs_configurations['cutoff']
    np.random.seed(srs_configurations['randomseed'])

    avg_vdegrees, avg_vneighs, std_vdegrees, std_vneighs = cd.simulation_cd(protocol, A, p_gen, q_swap, p_swap, p_cons, cutoff, max_links_swapped, qbits_per_channel, N_samples, total_time,
                  progress_bar=None, return_data='avg')

    return [avg_vdegrees, avg_vneighs, std_vdegrees, std_vneighs]


def virtual_cfs_info_avg(srs_configurations, N_samples, total_time, swap_mode):
    protocol = 'cfs'
    N_samples = N_samples
    total_time = total_time
    ## TOPOLOGY
    # Use any function main.adjacency_*() to define a topology.
    A = srs_configurations['adj']

    ## HARDWARE
    p_gen = srs_configurations['p_gen']  # Probability of successful entanglement generation
    p_swap = srs_configurations['p_swap']  # Probability of successful swap
    qbits_per_channel = srs_configurations['qubits']  # Number of qubits per node per physical neighbor

    ## SOFTWARE
    q_swap = srs_configurations['q_swap']  # Probability of performing swaps in the SRS protocol
    max_links_swapped = srs_configurations['max_swap']  #  Maximum number of elementary links swapped
    p_cons = srs_configurations['p_cons']  # Probability of virtual neighbors consuming a link per time step

    ## CUTOFF
    cutoff = srs_configurations['cutoff']
    # np.random.seed(srs_configurations['randomseed'])

    avg_vdegrees, avg_vneighs, std_vdegrees, std_vneighs = cd.simulation_cd_cfs(protocol, A, p_gen, q_swap, p_swap, p_cons, cutoff, max_links_swapped, qbits_per_channel, N_samples, total_time,
                  progress_bar=None, return_data='avg', swap_mode=swap_mode, randomseed=srs_configurations['randomseed'])

    return [avg_vdegrees, avg_vneighs, std_vdegrees, std_vneighs]

# def srs_config_squared_hard(q_swap, max_links_swapped, randomseed):
#     srs_configurations = {}
#
#     ## TOPOLOGY
#     # Use any function main.adjacency_*() to define a topology.
#     # Here we use a squared lattice (with hard boundary conditions)
#     # with 9 nodes as an example.
#     l = 3
#     n = int(l * l)
#     A = cd.adjacency_squared_hard(l)
#     topology = 'squared_hard'
#     srs_configurations['adj'] = A
#
#     ## HARDWARE
#     p_gen = 1  # Probability of successful entanglement generation
#     p_swap = 1  # Probability of successful swap
#     qbits_per_channel = 1  # Number of qubits per node per physical neighbor
#     srs_configurations['p_gen'] = p_gen
#     srs_configurations['p_swap'] = p_swap
#     srs_configurations['qubits'] = qbits_per_channel
#
#     ## SOFTWARE
#     # q_swap = 0.12  # Probability of performing swaps in the SRS protocol
#     # max_links_swapped = 4  #  Maximum number of elementary links swapped
#     p_cons = 0  # Probability of virtual neighbors consuming a link per time step
#     srs_configurations['q_swap'] = q_swap
#     srs_configurations['max_swap'] = max_links_swapped
#     srs_configurations['p_cons'] = p_cons
#
#     ## CUTOFF
#     cutoff = 20
#     srs_configurations['cutoff'] = cutoff
#     if randomseed is not None:
#         srs_configurations['randomseed'] = randomseed
#     else:
#         srs_configurations['randomseed'] = random.seed()
#
#     return srs_configurations


def batch_srs_info():
    randomseed = np.random.seed()
    qubit_per_channels = [2]
    cutoffs = [10]
    srs_info_list_avg = {}
    srs_info_list_vneighbor = {}
    N_samples = 5
    total_time = 1000
    path_write_avg = '../exp_data_pra/srs_info/srs_avg_2x2.json'
    path_write_vneighbors = '../exp_data_pra/srs_info/srs_vneighbors2x2.json'
    for i in range(6, 7):
        for qubit_per_channel in qubit_per_channels:
            for cutoff in cutoffs:
                randomseed = np.random.seed()
                srs_configurations = srs_config_squared_hard(q_swap=i * 0.02, qubit_per_channel=qubit_per_channel, p_gen = 1, p_swap = 1, p_cons = 0, cutoff=cutoff, randomseed=randomseed)
                para = "qswap_" + str(i) + "_cutoff_" + str(cutoff) + "_qubit_per_channel_" + str(qubit_per_channel)
                srs_info_list_avg[para] = virtual_srs_info_avg(srs_configurations, N_samples, total_time)
                srs_info_list_vneighbor[para] = virtual_srs_info_vneighbors(srs_configurations, N_samples, total_time)[2]
                # print(srs_info_list_avg[para])
                # for item in srs_info_list_avg[para]:
                #     print(item)
                # for item in srs_info_list_vneighbor[para]:
                #     print(item)
                # print(para, end=' ')
                # print("finished!")
    # json_str_srs_info_list_avg = json.dumps(srs_info_list_avg)
    # json_str_srs_info_list_vneighbor = json.dumps(srs_info_list_vneighbor)
    # with open(path_write_avg, 'w') as f:
    #     f.write(json_str_srs_info_list_avg)
    # with open(path_write_vneighbors, 'w') as f:
    #     f.write(json_str_srs_info_list_vneighbor)
    A = np.zeros((9, 9))
    # vneighborhoods = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
    for i, neighbor_list_all_node in enumerate(srs_info_list_vneighbor[para]):
        for sample in neighbor_list_all_node:
            for time in sample:
                for node in time:
                    if i == node:
                        continue
                    A[i][node] += 1
    print((A / (total_time * N_samples)).tolist())


def data_process_virtual_topo_matrix(path):
    # path = '../exp_data_pra_1127/protocol_data/cfs_vneighbors2x2.json'
    # json_str = ''
    with open(path, 'r') as f:
        line = f.readline()
        data_srs = json.loads(line)
    print(len(data_srs))
    data_channel1 = data_srs['qswap_6_cutoff_10_qubit_per_channel_1']
    # data_channel3 = data_srs['qswap_6_cutoff_10_qubit_per_channel_3']
    # data_channel5 = data_srs['qswap_6_cutoff_10_qubit_per_channel_5']
    A = np.zeros((9, 9))
    # vneighborhoods = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
    for i, neighbor_list_all_node in enumerate(data_channel1):
        for sample in neighbor_list_all_node:
            for time in sample:
                for node in time:
                    if i == node:
                        continue
                    A[i][node] += 1
    print((A / 10000).tolist())


def batch_cfs_info_test():
    randomseed = np.random.seed()
    # randomseed = 1
    # qubit_per_channels = [1, 3, 5]
    qubit_per_channels = [1]
    cutoffs = [i for i in range(6, 10)]
    srs_info_list_avg = {}
    srs_info_list_vneighbor = {}
    p_cons_list = [i * 0.05 for i in range(10)]
    p_swap_list = [i * 0.05 for i in range(10, 19)]
    q_swap_list = [i * 0.05 for i in range(6, 7)]
    N_samples = 1
    total_time = 10000
    swap_modes = ["total_distance", "algebraic_connectivity"]
    # swap_modes = ["total_distance"]
    path_write_avg = '../exp_data_pra/srs_info/srs_avg_3x3.json'
    path_write_vneighbors = '../exp_data_pra/srs_info/srs_vneighbors3x3.json'
    for q_swap in q_swap_list:
        for qubit_per_channel in qubit_per_channels:
            for p_cons in p_cons_list:
                for cutoff in cutoffs:
                    for p_swap in p_swap_list:
                        for swap_mode in swap_modes:
                            # randomseed = 1
                            srs_configurations = srs_config_squared_hard(q_swap=q_swap,
                                                                         qubit_per_channel=qubit_per_channel, p_gen=1,
                                                                         p_swap=p_swap, p_cons=p_cons, cutoff=cutoff,
                                                                         randomseed=None)
                            # qubit_per_channel, p_gen, p_swap, q_swap, p_cons, cutoff, randomseed
                            para = f"{swap_mode}_qswap_{q_swap:.2f}_qubit_per_channel_{qubit_per_channel}_p_swap{p_swap:.2f}_p_cons{p_cons:.2f}_cutoff_{cutoff}.npy"

                            srs_info_list_avg[para] = virtual_cfs_info_avg(srs_configurations, N_samples, total_time,
                                                                           swap_mode)
                            srs_info_list_vneighbor[para] = \
                            virtual_cfs_info_vneighbors(srs_configurations, N_samples, total_time, swap_mode)[2]
                            A = np.zeros((9, 9))
                            # vneighborhoods = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
                            for i, neighbor_list_all_node in enumerate(srs_info_list_vneighbor[para]):
                                for sample in neighbor_list_all_node:
                                    for time in sample:
                                        for node in time:
                                            if i == node:
                                                continue
                                            A[i][node] += 1
                            virtual_adjacency_matrix = A / (total_time * N_samples)
                            # print((A / (total_time * N_samples)).tolist())
                            matrix_path = os.path.join("/home/normaluser/hflash/qnet_iwqos/prototol_virtual_matrix_data", para)
                            np.save(matrix_path, virtual_adjacency_matrix)
                            # print(matrix_path)
                # for item in srs_info_list_avg[para]:
                #     print(item)
                # for item in srs_info_list_vneighbor[para]:
                #     print(item)
                # print(para, end=' ')
                # print("finished!")
    # json_str_srs_info_list_avg = json.dumps(srs_info_list_avg)
    # json_str_srs_info_list_vneighbor = json.dumps(srs_info_list_vneighbor)
    # with open(path_write_avg, 'w') as f:
    #     f.write(json_str_srs_info_list_avg)
    # with open(path_write_vneighbors, 'w') as f:
    #     f.write(json_str_srs_info_list_vneighbor)
    # A = np.zeros((9, 9))
    # vneighborhoods = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
    # for i, neighbor_list_all_node in enumerate(srs_info_list_vneighbor[para]):
    #     for sample in neighbor_list_all_node:
    #         for time in sample:
    #             for node in time:
    #                 if i == node:
    #                     continue
    #                 A[i][node] += 1
    # print((A / (total_time * N_samples)).tolist())

# import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor

def process_params(q_swap, qubit_per_channel, p_cons, cutoff, p_swap, swap_mode, N_samples, total_time):
    # Generate SRS configurations
    srs_configurations = srs_config_squared_hard(q_swap=q_swap,
                                                  qubit_per_channel=qubit_per_channel, p_gen=1,
                                                  p_swap=p_swap, p_cons=p_cons, cutoff=cutoff,
                                                  randomseed=None)

    para = f"{swap_mode}_qswap_{q_swap:.2f}_qubit_per_channel_{qubit_per_channel}_p_swap{p_swap:.2f}_p_cons{p_cons:.2f}_cutoff_{cutoff}.npy"
    print(para)
    matrix_path = os.path.join("/home/normaluser/hflash/qnet_iwqos/prototol_virtual_matrix_data", para)
    # np.save(matrix_path, virtual_adjacency_matrix)
    if os.path.exists(matrix_path):
        return para
    # srs_info_avg = virtual_cfs_info_avg(srs_configurations, N_samples, total_time, swap_mode)
    srs_info_vneighbor = virtual_cfs_info_vneighbors(srs_configurations, N_samples, total_time, swap_mode)[2]
    A = np.zeros((9, 9))

    for i, neighbor_list_all_node in enumerate(srs_info_vneighbor):
        for sample in neighbor_list_all_node:
            for time in sample:
                for node in time:
                    if i == node:
                        continue
                    A[i][node] += 1

    virtual_adjacency_matrix = A / (total_time * N_samples)
    matrix_path = os.path.join("/home/normaluser/hflash/qnet_iwqos/prototol_virtual_matrix_data", para)
    np.save(matrix_path, virtual_adjacency_matrix)
    print("finished!", para)
    # print(para)
    return para  # Return the parameter string for tracking


def process_params_chain(q_swap, qubit_per_channel, p_cons, cutoff, p_swap, swap_mode, N_samples, total_time):
    # Generate SRS configurations
    srs_configurations = srs_config_chain(q_swap=q_swap,
                                                  qubit_per_channel=qubit_per_channel, p_gen=1,
                                                  p_swap=p_swap, p_cons=p_cons, cutoff=cutoff,
                                                  randomseed=None)

    para = f"{swap_mode}_qswap_{q_swap:.2f}_qubit_per_channel_{qubit_per_channel}_p_swap{p_swap:.2f}_p_cons{p_cons:.2f}_cutoff_{cutoff}.npy"
    print(para)
    # srs_info_avg = virtual_cfs_info_avg(srs_configurations, N_samples, total_time, swap_mode)
    srs_info_vneighbor = virtual_cfs_info_vneighbors(srs_configurations, N_samples, total_time, swap_mode)[2]

    A = np.zeros((9, 9))

    for i, neighbor_list_all_node in enumerate(srs_info_vneighbor):
        for sample in neighbor_list_all_node:
            for time in sample:
                for node in time:
                    if i == node:
                        continue
                    A[i][node] += 1

    virtual_adjacency_matrix = A / (total_time * N_samples)

    matrix_path = os.path.join("/home/normaluser/hflash/qnet_iwqos/prototol_virtual_matrix_data_chain", para)
    # print("write path: ", matrix_path)
    np.save(matrix_path, virtual_adjacency_matrix)
    print("finished!", para)
    # print(para)
    return para  # Return the parameter string for tracking





def parallel_sampling(max_workers):
    np.random.seed()
    qubit_per_channels = [1, 3, 5]
    cutoffs = [i for i in range(6, 11)]
    p_cons_list = [i * 0.05 for i in range(0, 11)]
    p_swap_list = [i * 0.05 for i in range(10, 20)]
    q_swap_list = [i * 0.05 for i in range(6, 7)]
    N_samples = 1
    total_time = 1000
    swap_modes = ["total_distance", "algebraic_connectivity"]

    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        futures = []
        for q_swap in q_swap_list:
            for qubit_per_channel in qubit_per_channels:
                for p_cons in p_cons_list:
                    for cutoff in cutoffs:
                        for p_swap in p_swap_list:
                            for swap_mode in swap_modes:
                                # Submit a new task
                                futures.append(executor.submit(process_params_chain, q_swap, qubit_per_channel,
                                                               p_cons, cutoff, p_swap, swap_mode,
                                                               N_samples, total_time))

        # Optional: Handle results
        for future in futures:
            print("Processed:", future.result())


if __name__ == '__main__':
    parallel_sampling(100)

    # batch_srs_info()
    # path = '../prototol_virtual_matrix_data/'

