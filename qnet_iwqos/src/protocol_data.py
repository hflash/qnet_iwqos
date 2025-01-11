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
# @Time     : 2025/1/6 9:12
# @Author   : HFLASH @ LINKE
# @File     : protocol_data.py
# @Software : PyCharm

import os.path
import numpy as np


def get_cfs_virtual_adjacency_matrix_of_grid(qubit_per_channel, cutoff, p_cons,
                                              p_swap,
                                              q_swap,
                                              swap_mode):
    para = f"{swap_mode}_qswap_{q_swap:.2f}_qubit_per_channel_{qubit_per_channel}_p_swap{p_swap:.2f}_p_cons{p_cons:.2f}_cutoff_{cutoff}.npy"
    root_path = "/home/normaluser/hflash/qnet_iwqos/prototol_virtual_matrix_data"
    data_path = os.path.join(root_path, para)
    if os.path.exists(data_path):
        data_matrix = np.load(data_path)
        return data_matrix
    else:
        print(f"{data_path}\nPath not exists!")

def get_cfs_virtual_adjacency_matrix_of_chain(qubit_per_channel, cutoff, p_cons,
                                              p_swap,
                                              q_swap,
                                              swap_mode):
    para = f"{swap_mode}_qswap_{q_swap:.2f}_qubit_per_channel_{qubit_per_channel}_p_swap{p_swap:.2f}_p_cons{p_cons:.2f}_cutoff_{cutoff}.npy"
    root_path = "/home/normaluser/hflash/qnet_iwqos/prototol_virtual_matrix_data_chain"
    data_path = os.path.join(root_path, para)
    print(data_path)
    if os.path.exists(data_path):
        data_matrix = np.load(data_path)
        return data_matrix
    else:
        print("Path not exists!")


def get_data_by_path(protocol):
    # data_path = "/home/normaluser/hflash/qnet_iwqos/prototol_virtual_matrix_data/random_qswap_0.12_qubit_per_channel_hetero_random_p_swap0.95_p_cons0.05_cutoff_10.npy"
    data_path = ''
    if protocol == 'srs':
        data_path = f'/home/normaluser/hflash/qnet_iwqos/test_data/{protocol}/all_links_random_qswap_0.12_qubit_per_channel_10_p_swap0.95_p_cons0.05_cutoff_10.npy'
    else:
        data_path = "/home/normaluser/hflash/qnet_iwqos/test_data/cfs/all_links_algebraic_connectivity_qswap_0.12_qubit_per_channel_10_p_swap0.95_p_cons0.05_cutoff_10.npy"
    if os.path.exists(data_path):
        data_matrix = np.load(data_path)
        return data_matrix
    else:
        print("Path not exists!")

# print(get_cfs_virtual_adjacency_matrix_of_grid(qubit_per_channel = 3, cutoff = 7, p_cons = 0.35,
#                                               p_swap = 0.95,
#                                               q_swap = 0.30,
#                                               swap_mode = "algebraic_connectivity"))
# print(get_cfs_virtual_adjacency_matrix_of_chain(qubit_per_channel = 1, cutoff = 10, p_cons = 0.05,
#                                               p_swap = 0.50,
#                                               q_swap = 0.30,
#                                               swap_mode = "algebraic_connectivity"))

def protocol_data_batch():
    qubit_per_channels = [1, 3, 5]
    cutoffs = [i for i in range(6, 11)]
    p_cons_list = [i * 0.05 for i in range(0, 11)]
    p_swap_list = [i * 0.05 for i in range(10, 20)]
    q_swap_list = [i * 0.05 for i in range(6, 7)]
    # N_samples = 1
    # total_time = 1000
    swap_modes = ["total_distance", "algebraic_connectivity"]
    count = 0
    correct_count = 0
    for q_swap in q_swap_list:
        for qubit_per_channel in qubit_per_channels:
            for p_cons in p_cons_list:
                for cutoff in cutoffs:
                    for p_swap in p_swap_list:
                        for swap_mode in swap_modes:
                            para = f"{swap_mode}_qswap_{q_swap:.2f}_qubit_per_channel_{qubit_per_channel}_p_swap{p_swap:.2f}_p_cons{p_cons:.2f}_cutoff_{cutoff}.npy"
                            matrix_path = os.path.join(
                                "/home/normaluser/hflash/qnet_iwqos/srs_prototol_virtual_matrix_data_grid", para)
                            if not os.path.exists(matrix_path):
                                print(para)
                                count += 1
                                continue
                            correct_count += 1
                            print(np.load(matrix_path))
    return correct_count, count



if __name__ == "__main__":
    print(get_data_by_path('srs'))
    print(get_data_by_path('cfs'))