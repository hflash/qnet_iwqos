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
# @Time     : 2024/2/22 23:05
# @Author   : HFLASH @ LINKE
# @File     : srs_data.py
# @Software : PyCharm
import os.path

import numpy as np


def get_virtual_adjacency_matrix_of_3x3(qubit_per_channel):
    if qubit_per_channel == 1:
        return np.array([[0.0, 0.5064, 0.0506, 0.5153, 0.1713, 0.0525, 0.0516, 0.0443, 0.0136],
                         [0.5064, 0.0, 0.5097, 0.2346, 0.5793, 0.2304, 0.0424, 0.0798, 0.046],
                         [0.0506, 0.5097, 0.0, 0.0517, 0.1861, 0.5236, 0.0112, 0.0453, 0.0466],
                         [0.5153, 0.2346, 0.0517, 0.0, 0.5864, 0.0802, 0.5031, 0.2664, 0.0338],
                         [0.1713, 0.5793, 0.1861, 0.5864, 0.0, 0.5972, 0.1788, 0.5842, 0.1656],
                         [0.0525, 0.2304, 0.5236, 0.0802, 0.5972, 0.0, 0.0413, 0.2337, 0.5342],
                         [0.0516, 0.0424, 0.0112, 0.5031, 0.1788, 0.0413, 0.0, 0.4943, 0.0522],
                         [0.0443, 0.0798, 0.0453, 0.2664, 0.5842, 0.2337, 0.4943, 0.0, 0.5153],
                         [0.0136, 0.046, 0.0466, 0.0338, 0.1656, 0.5342, 0.0522, 0.5153, 0.0]])
    if qubit_per_channel == 3:
        return np.array([[0.0, 0.9804, 0.1225, 0.9791, 0.2669, 0.0402, 0.1136, 0.0394, 0.004],
                         [0.9804, 0.0, 0.9782, 0.3852, 0.9906, 0.3902, 0.0421, 0.0855, 0.0391],
                         [0.1225, 0.9782, 0.0, 0.0391, 0.2874, 0.977, 0.0041, 0.0481, 0.1041],
                         [0.9791, 0.3852, 0.0391, 0.0, 0.9913, 0.0878, 0.9796, 0.409, 0.0314],
                         [0.2669, 0.9906, 0.2874, 0.9913, 0.0, 0.9922, 0.2678, 0.9925, 0.2431],
                         [0.0402, 0.3902, 0.977, 0.0878, 0.9922, 0.0, 0.0397, 0.4109, 0.9747],
                         [0.1136, 0.0421, 0.0041, 0.9796, 0.2678, 0.0397, 0.0, 0.9798, 0.1033],
                         [0.0394, 0.0855, 0.0481, 0.409, 0.9925, 0.4109, 0.9798, 0.0, 0.9711],
                         [0.004, 0.0391, 0.1041, 0.0314, 0.2431, 0.9747, 0.1033, 0.9711, 0.0]])
    if qubit_per_channel == 5:
        return np.array([[0.0, 0.9976, 0.1106, 0.9974, 0.2716, 0.0346, 0.1207, 0.0276, 0.0012],
                         [0.9976, 0.0, 0.9961, 0.4231, 0.999, 0.4225, 0.029, 0.0807, 0.0212],
                         [0.1106, 0.9961, 0.0, 0.0251, 0.2689, 0.9969, 0.0016, 0.0293, 0.1267],
                         [0.9974, 0.4231, 0.0251, 0.0, 0.9986, 0.0813, 0.9972, 0.4182, 0.0261],
                         [0.2716, 0.999, 0.2689, 0.9986, 0.0, 0.9989, 0.2613, 0.9982, 0.2627],
                         [0.0346, 0.4225, 0.9969, 0.0813, 0.9989, 0.0, 0.028, 0.4012, 0.9981],
                         [0.1207, 0.029, 0.0016, 0.9972, 0.2613, 0.028, 0.0, 0.9963, 0.1347],
                         [0.0276, 0.0807, 0.0293, 0.4182, 0.9982, 0.4012, 0.9963, 0.0, 0.9977],
                         [0.0012, 0.0212, 0.1267, 0.0261, 0.2627, 0.9981, 0.1347, 0.9977, 0.0]])

def get_cfs_virtual_adjacency_matrix_of_chain(qubit_per_channel, cutoff, p_cons,
    p_swap,
    q_swap,
    swap_mode):
    para = f"{swap_mode}_qswap_{q_swap:.2f}_qubit_per_channel_{qubit_per_channel}_p_swap{p_swap:.2f}_p_cons{p_cons:.2f}_cutoff_{cutoff}.npy"
    root_path = "/home/normaluser/hflash/qnet_iwqos/prototol_virtual_matrix_data_chain"
    data_path = os.path.join(root_path, para)
    if not os.path.exists(data_path):
        data_matrix = np.load(data_path)
    return data_matrix

