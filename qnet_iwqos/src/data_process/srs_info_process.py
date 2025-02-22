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
# @Time     : 2024/2/22 16:35
# @Author   : HFLASH @ LINKE
# @File     : srs_info_process.py
# @Software : PyCharm
import json

import numpy as np

if __name__ == '__main__':
    path = './srs_vneighbors2x2.json'
    # json_str = ''
    with open(path, 'r') as f:
        line = f.readline()
        data_srs = json.loads(line)
    print(len(data_srs))
    data_channel1 = data_srs['qswap_6_cutoff_10_qubit_per_channel_1']
    # data_channel3 = data_srs['qswap_6_cutoff_10_qubit_per_channel_3']
    # data_channel5 = data_srs['qswap_6_cutoff_10_qubit_per_channel_5']
    A = np.zeros((4, 4))
    # vneighborhoods = [[[None for _ in range(N_samples)] for _ in range(total_time)] for _ in range(n)]
    for i, neighbor_list_all_node in enumerate(data_channel1):
        for sample in neighbor_list_all_node:
            for time in sample:
                for node in time:
                    if i == node:
                        continue
                    A[i][node] += 1
    print((A/10000).tolist())
