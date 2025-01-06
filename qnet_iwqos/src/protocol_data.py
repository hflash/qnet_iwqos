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


def get_cfs_virtual_adjacency_matrix_of_chain(qubit_per_channel, cutoff, p_cons,
    p_swap,
    q_swap,
    swap_mode):
    para = f"{swap_mode}_qswap_{q_swap:.2f}_qubit_per_channel_{qubit_per_channel}_p_swap{p_swap:.2f}_p_cons{p_cons:.2f}_cutoff_{cutoff}.npy"
    root_path = "/home/normaluser/hflash/qnet_iwqos/prototol_virtual_matrix_data"
    data_path = os.path.join(root_path, para)
    if not os.path.exists(data_path):
        data_matrix = np.load(data_path)
        return data_matrix
    else:
        print("Path not exists!")


