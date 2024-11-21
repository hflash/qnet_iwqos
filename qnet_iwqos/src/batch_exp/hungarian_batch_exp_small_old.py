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
# @Time     : 2024/2/21 14:35
# @Author   : HFLASH @ LINKE
# @File     : batch_exp_small.py
# @Software : PyCharm
import os
import time
import numpy as np
import distributed_operation_circuit as dioc
from circuit2graph import circuitPartition
from srs_data import get_virtual_adjacency_matrix_of_3x3


# L1: 线路分割+线路映射
# L2: 线路执行优化
def batch_circuit_execution(schedule, qswap, cutoff, qubit_per_channel, num_sample, small_device_qubit_number, large_device_qubit_number):
    # todo: change the file path
    path = '../exp_circuit_benchmark/small_scale'
    # path = './'
    # print(circuitPartition(path))
    # data = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.qasm'):
                qasm_path = os.path.join(root, file)
                circuitname = file.split('.qasm')[-2]
                print(qasm_path)
                # device_qubit_number = 2
                # if "cm42" in qasm_path or "cm82" in qasm_path or "z4" in qasm_path:
                #     continue
                # if "cm82" not in qasm_path:
                #     continue
                # assert 'small' in qasm_path or 'large' in qasm_path
                if 'small' in qasm_path:
                    scale = 'small'
                    device_qubit_number = small_device_qubit_number
                if 'large' in qasm_path:
                    scale = 'large'
                    device_qubit_number = large_device_qubit_number
                for i in range(num_sample):
                    # todo: change the write file path
                    path_write = '../exp_data/ours_0222_all_time/' + scale + '/' + circuitname + '_shortest_path_' + "hungarian_allocation" + "_bandwidth_" + str(qubit_per_channel) + '_old' + '.txt'
                    print(path_write)
                    randomseed = np.random.seed()
                    # randomseed = 0
                    timestart = time.time()
                    remote_operations, circuit_dagtable, gate_list, subcircuits_communication, qubit_loc_subcircuit_dic, subcircuit_qubit_partitions = circuitPartition(
                        qasm_path, device_qubit_number, randomseed)
                    srs_configurations = dioc.srs_config_squared_hard(qubit_per_channel, qswap, cutoff, randomseed)
                    # srs_configurations = dioc.srs_config_tree(qubit_per_channel, cutoff, randomseed)
                    # srs_configurations = dioc.srs_config_chain(qubit_per_channel, cutoff, randomseed)
                    # todo: change the allocate method
                    # subcircuits_allocation = dioc.trivial_allocate_subcircuit(len(subcircuit_qubit_partitions),
                    #                                                      subcircuits_communication,
                    #                                                      srs_configurations['adj'])
                    subcircuits_allocation = dioc.Hungarian_allocate_subcircuit(len(subcircuit_qubit_partitions),
                                                                                subcircuits_communication,
                                                                                get_virtual_adjacency_matrix_of_3x3(qubit_per_channel))
                    timestart = time.time()
                    # todo 修改这个方法
                    ecost, time_step, execution_schedule, discard_entanglement_count = dioc.time_evolution_old(
                        srs_configurations, circuit_dagtable,
                        gate_list, qubit_loc_subcircuit_dic,
                        subcircuits_allocation, remote_operations,
                        schedule)
                    process_time = time.time() - timestart
                    print(ecost, time_step)
                    print(process_time)
                    print(discard_entanglement_count)
                    # data.append([ecost, time_step, discard_entanglement_count, process_time])
                    with open(path_write, 'a+') as f:
                        f.write("Sample: " + str(i))
                        f.write('\n')
                        f.write("Time cost: " + str(time_step))
                        f.write('\n')
                        f.write("Entanglement cost:" + str(ecost))
                        f.write('\n')
                        f.write("Processing time: " + str(process_time))
                        f.write('\n')
                        f.write("Discard entanglements: " + str(discard_entanglement_count))
                        f.write('\n')
                        print("Write Succeed!")


if __name__ == "__main__":
    # schedules = [0, 1]
    schedules = [0]
    small_device_qubit_number = 5
    large_device_qubit_number = 40
    qubit_per_channels = [1, 3, 5]
    # qubit_per_channels = [3]
    N_samples = 20
    # data = []
    # data_cal = 0
    # cutoffs = [i for i in range(1, 11)]
    # q_swaps = [i * 0.1 for i in range(1, 11)]
    for schedule in schedules:
        for qubit_per_channel in qubit_per_channels:
            batch_circuit_execution(schedule, 0.12, 10, qubit_per_channel, N_samples, small_device_qubit_number, large_device_qubit_number)
