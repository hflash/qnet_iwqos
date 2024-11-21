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


# L1: 线路分割+线路映射
# L2: 线路执行优化
def batch_circuit_execution(schedule, qswap, cutoff, qubit_per_channel, num_sample, small_device_qubit_number, large_device_qubit_number):
    path = '../trial_bench/medium/z4'
    # path = './'
    # print(circuitPartition(path))
    phsical_adjacency_matrix = np.array([[0.0, 0.5064, 0.0506, 0.5153, 0.1713, 0.0525, 0.0516, 0.0443, 0.0136],
     [0.5064, 0.0, 0.5097, 0.2346, 0.5793, 0.2304, 0.0424, 0.0798, 0.046],
     [0.0506, 0.5097, 0.0, 0.0517, 0.1861, 0.5236, 0.0112, 0.0453, 0.0466],
     [0.5153, 0.2346, 0.0517, 0.0, 0.5864, 0.0802, 0.5031, 0.2664, 0.0338],
     [0.1713, 0.5793, 0.1861, 0.5864, 0.0, 0.5972, 0.1788, 0.5842, 0.1656],
     [0.0525, 0.2304, 0.5236, 0.0802, 0.5972, 0.0, 0.0413, 0.2337, 0.5342],
     [0.0516, 0.0424, 0.0112, 0.5031, 0.1788, 0.0413, 0.0, 0.4943, 0.0522],
     [0.0443, 0.0798, 0.0453, 0.2664, 0.5842, 0.2337, 0.4943, 0.0, 0.5153],
     [0.0136, 0.046, 0.0466, 0.0338, 0.1656, 0.5342, 0.0522, 0.5153, 0.0]])
    # phsical_adjacency_matrix = np.array([[0.0, 0.9804, 0.1225, 0.9791, 0.2669, 0.0402, 0.1136, 0.0394, 0.004], [0.9804, 0.0, 0.9782, 0.3852, 0.9906, 0.3902, 0.0421, 0.0855, 0.0391], [0.1225, 0.9782, 0.0, 0.0391, 0.2874, 0.977, 0.0041, 0.0481, 0.1041], [0.9791, 0.3852, 0.0391, 0.0, 0.9913, 0.0878, 0.9796, 0.409, 0.0314], [0.2669, 0.9906, 0.2874, 0.9913, 0.0, 0.9922, 0.2678, 0.9925, 0.2431], [0.0402, 0.3902, 0.977, 0.0878, 0.9922, 0.0, 0.0397, 0.4109, 0.9747], [0.1136, 0.0421, 0.0041, 0.9796, 0.2678, 0.0397, 0.0, 0.9798, 0.1033], [0.0394, 0.0855, 0.0481, 0.409, 0.9925, 0.4109, 0.9798, 0.0, 0.9711], [0.004, 0.0391, 0.1041, 0.0314, 0.2431, 0.9747, 0.1033, 0.9711, 0.0]])

    data = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.qasm'):
                qasm_path = os.path.join(root, file)
                circuitname = file.split('.qasm')[-2]
                print(qasm_path)
                device_qubit_number = 5
                # if "cm42" in qasm_path or "cm82" in qasm_path or "z4" in qasm_path:
                #     continue
                # assert 'small' in qasm_path or 'large' in qasm_path
                # if 'small' in qasm_path:
                #     scale = 'small'
                #     device_qubit_number = small_device_qubit_number
                # if 'large' in qasm_path:
                #     scale = 'large'
                #     device_qubit_number = large_device_qubit_number
                for i in range(2):
                    # path_write = '../exp_data/baseline_0221_unweighted/' + scale + '/' + circuitname + '_schedule_' + str(schedule) + '_sample_' + str(
                    #     i) + "_trivial_allocation" + "_bandwidth_2" + '.qasm'
                    # print(path_write)
                    randomseed = np.random.seed()
                    # randomseed = 0
                    timestart = time.time()
                    remote_operations, circuit_dagtable, gate_list, subcircuits_communication, qubit_loc_subcircuit_dic, subcircuit_qubit_partitions = circuitPartition(
                        qasm_path, device_qubit_number, randomseed)
                    srs_configurations = dioc.srs_config_squared_hard(qubit_per_channel, qswap, cutoff, randomseed)
                    # srs_configurations = dioc.srs_config_tree(qubit_per_channel, cutoff, randomseed)
                    # srs_configurations = dioc.srs_config_chain(qubit_per_channel, cutoff, randomseed)
                    # subcircuits_allocation = dioc.trivial_allocate_subcircuit(len(subcircuit_qubit_partitions),
                    #                                                      subcircuits_communication,
                    #                                                      srs_configurations['adj'])
                    # print(subcircuits_allocation)
                    subcircuits_allocation = dioc.Hungarian_allocate_subcircuit(len(subcircuit_qubit_partitions),
                                                                         subcircuits_communication, phsical_adjacency_matrix)
                    # print(subcircuits_allocation)
                    # ecost, time_step, execution_schedule, discard_entanglement_count = dioc.time_evolution(srs_configurations, circuit_dagtable,
                    #                                                       gate_list, qubit_loc_subcircuit_dic,
                    #                                                       subcircuits_allocation, remote_operations,
                    #                                                       schedule)
                    # process_time = time.time() - timestart
                    # data.append([ecost, time_step, discard_entanglement_count, process_time])
                    # print(ecost, time_step)
                    # print(process_time)
                    # print(discard_entanglement_count)
                    # with open(path_write, 'a+') as f:
                    #     f.write("qubit location dic: " + str(qubit_loc_subcircuit_dic))
                    #     f.write('\n')
                    #     f.write("subcircuit qubit partitions: " + str(subcircuit_qubit_partitions))
                    #     f.write('\n')
                    #     f.write("subcircuits allocation" + str(subcircuits_allocation))
                    #     f.write('\n')
                    #     f.write("Time cost: " + str(time_step))
                    #     f.write('\n')
                    #     f.write("Entanglement cost:" + str(ecost))
                    #     f.write('\n')
                    #     f.write("Processing time: " + str(process_time))
                    #     print("Write Succeed!")
                    # break
                    # print(qubit_loc_subcircuit_dic, subcircuit_qubit_partitions)
                    # print(subcircuits_allocation)


                    # only remote
                    # timestart = time.time()
                    # ecost, time_step, execution_schedule, discard_entanglement_count = dioc.time_evolution_only_remote_time(
                    #     srs_configurations, circuit_dagtable,
                    #     gate_list, qubit_loc_subcircuit_dic,
                    #     subcircuits_allocation, remote_operations,
                    #     schedule)
                    # process_time = time.time() - timestart
                    # # # print(ecost, time_step)
                    # # # print(process_time)
                    # # # print(discard_entanglement_count)
                    # data.append([ecost, time_step, discard_entanglement_count, process_time])
                    #
                    # # old
                    timestart = time.time()
                    ecost, time_step, execution_schedule, discard_entanglement_count = dioc.time_evolution_old(
                        srs_configurations, circuit_dagtable,
                        gate_list, qubit_loc_subcircuit_dic,
                        subcircuits_allocation, remote_operations,
                        schedule)
                    process_time = time.time() - timestart
                    # print(ecost, time_step)
                    # print(process_time)
                    # print(discard_entanglement_count)
                    data.append([ecost, time_step, discard_entanglement_count, process_time])
                    #
                    # # old only remote
                    # timestart = time.time()
                    # ecost, time_step, execution_schedule, discard_entanglement_count = dioc.time_evolution_old_only_remote_time(
                    #     srs_configurations, circuit_dagtable,
                    #     gate_list, qubit_loc_subcircuit_dic,
                    #     subcircuits_allocation, remote_operations,
                    #     schedule)
                    # process_time = time.time() - timestart
                    # # print(ecost, time_step)
                    # # print(process_time)
                    # # print(discard_entanglement_count)
                    # data.append([ecost, time_step, discard_entanglement_count, process_time])
            # break
    return data


if __name__ == "__main__":
    # schedules = [0, 1]
    schedules = [0]
    small_device_qubit_number = 5
    large_device_qubit_number = 40
    # qubit_per_channels = [1, 3, 5]
    qubit_per_channels = [1]
    N_samples = 20
    data = []
    data_cal = 0
    cutoffs = [10]
    q_swaps = [0.12]
    for schedule in schedules:
        for qubit_per_channel in qubit_per_channels:
            for qswap in q_swaps:
                data_cal_ecost = 0
                data_cal_time = 0
                data_cal_discard = 0
                data_cal_ecost_old = 0
                data_cal_time_old = 0
                data_cal_discard_old = 0
                result = batch_circuit_execution(schedule, 0.12, 10, qubit_per_channel, N_samples, small_device_qubit_number, large_device_qubit_number)
                for index, sample in enumerate(result):
                    # if index % 2 == 0:
                    data_cal_ecost += sample[0]
                    data_cal_time += sample[1]
                    data_cal_discard += sample[2]
                    # else:
                    #     data_cal_ecost_old += sample[0]
                    #     data_cal_time_old += sample[1]
                    #     data_cal_discard_old += sample[2]
                print("--------------schedule %s--------------"% str(schedule))
                print("--------------qubit_per_channel %s--------------" % str(qubit_per_channel))
                print("--------------cutoff %s--------------" % str(10))
                print("--------------q_swap %s--------------" % str(qswap))
                # print("--------------ecost--------------")
                data.append([data_cal_ecost/N_samples, data_cal_time/N_samples, data_cal_discard/N_samples])
                print(data_cal_ecost/N_samples, data_cal_time/N_samples, data_cal_discard/N_samples)
                # print(data_cal_ecost_old / N_samples)
                # print("--------------time--------------")
                # print(data_cal_time/N_samples)
                # print(data_cal_time_old / N_samples)
                # print("--------------discard--------------")
                # print(data_cal_discard/N_samples)
                # print(data_cal_discard_old/N_samples)
    print(data)