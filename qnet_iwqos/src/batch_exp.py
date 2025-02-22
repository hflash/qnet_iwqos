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
import pandas as pd
import distributed_operation_circuit as dioc
from multiprocessing import Pool, pool, Lock
from circuit2graph import circuitPartition
from protocol_data import get_cfs_virtual_adjacency_matrix_of_chain, get_cfs_virtual_adjacency_matrix_of_grid, \
    get_data_by_path
from srs_data import get_virtual_adjacency_matrix_of_3x3
import re
import math
from quantumcircuit.remotedag import RemoteDag
from avarage_physical_distance import physical_avg_dist_grid
from matrix2matrix import compute_mapping_information, cost_matrix, cost_matrix_sliced


class Task:
    def __init__(self, circuit_path, name, paras: dict, save_path):
        self.lock = Lock()
        self.circuit_path = circuit_path
        self.circuit_name = name
        self.p_gen = paras['p_gen']
        self.p_swap = paras['p_swap']
        self.q_swap = paras['q_swap']
        self.p_cons = paras['p_cons']
        self.cutoff = paras['cutoff']
        self.swap_mode = paras['swap_mode']
        self.allocation = paras['allocation']
        self.channels = paras['channels']
        self.schedule = paras['schedule']
        self.sample_num = paras['samples']

        self.data = []
        self.has_done = 0
        self.data_path = os.path.join(save_path, '{}_{}_{:.2f}'.format(name, self.channels, self.p_cons))

    def call_back(self, result):
        with self.lock:
            self.data.append(result)
            self.has_done += 1
            if self.has_done == self.sample_num:
                # self.dump_data()
                pass

    def dump_data(self):
        with open(self.data_path, 'w') as f:
            f.write('time cost \t\t\t entanglement \t\t\t costprocessing time \t\t\t discard entanglements\n')
            for row in self.data:
                f.write('{:<9} \t\t\t {:<12} \t\t\t {:<19} \t\t\t {}\n'.format(row[0], row[1], row[2], row[3]))

    def get_data(self):
        return self.data


def run_parallel(tasks_list: list[list[Task]], small_device_qubit_number, large_device_qubit_number):
    p = Pool()
    for tasks in tasks_list:
        for task in tasks:
            for _ in range(task.sample_num):
                p.apply_async(circuit_execution,
                              (task.circuit_path, task.p_gen, task.p_swap, task.q_swap, task.p_cons, task.cutoff,
                               task.swap_mode, task.allocation, task.channels, task.schedule, small_device_qubit_number,
                               large_device_qubit_number),
                              callback=task.call_back)

    p.close()
    p.join()


# L1: 线路分割+线路映射
# L2: 线路执行优化
def circuit_execution(qasm_path, p_gen, p_swap, q_swap, p_cons, cutoff, swap_mode, allocation, channels, schedule,
                      small_device_qubit_number, large_device_qubit_number):
    # if 'small' in qasm_path:
    #     scale = 'small'
    #     device_qubit_number = small_device_qubit_number
    # else:
    #     scale = 'large'
    #     device_qubit_number = large_device_qubit_number
    randomseed = np.random.seed()

    device_qubit_number = 0
    qubit_number_pattern = re.compile('\S+?_n?(\d+)\S*?\.qasm')
    match_obj = re.match(qubit_number_pattern, qasm_path)
    if match_obj:
        qubit_number = int(match_obj.group(1))
        device_qubit_number = math.ceil(qubit_number / 9)

    assert (device_qubit_number > 0)
    # randomseed = 0
    timestart = time.time()
    remote_operations, circuit_dagtable, gate_list, subcircuits_communication, qubit_loc_subcircuit_dic, subcircuit_qubit_partitions = circuitPartition(
        qasm_path, device_qubit_number, randomseed)
    qubit_cnt = len(circuit_dagtable)
    physical_bandwidth = [[0.0, 4, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0], [4, 0.0, 5, 0.0, 4, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 5, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0], [1, 0.0, 0.0, 0.0, 1, 0.0, 1, 0.0, 0.0],
                          [0.0, 4, 0.0, 1, 0.0, 5, 0.0, 4, 0.0], [0.0, 0.0, 1, 0.0, 5, 0.0, 0.0, 0.0, 1],
                          [0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 4, 0.0], [0.0, 0.0, 0.0, 0.0, 4, 0.0, 4, 0.0, 1],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 1, 0.0]]
    srs_configurations = dioc.srs_config_squared_hard(qubit_per_channel=physical_bandwidth, p_gen=p_gen, p_swap=p_swap,
                                                      q_swap=q_swap, p_cons=p_cons, cutoff=cutoff,
                                                      randomseed=randomseed)
    # max_value = np.sum(subcircuits_communication) * 2 / np.sum(physical_bandwidth)
    # normalized_subcircuits_communication = dioc.normalize_subcircuit_communication(max_value, subcircuits_communication)
    remote_dag = RemoteDag(qubit_cnt, remote_operations, gate_list, qubit_loc_subcircuit_dic)
    slice_depth = 1
    # sliced_subcircuit_communication = dioc.sliced_subcircuit_communication_generate(remote_dag, slice_depth,
    #                                                                                 len(physical_bandwidth))

    # max_value = np.sum(subcircuits_communication) * 2 / np.sum(physical_bandwidth)
    max_value = 600
    normalized_subcircuits_communication = dioc.normalize_subcircuit_communication(max_value, subcircuits_communication)
    costmatrix = cost_matrix(normalized_subcircuits_communication, get_data_by_path('cfs'))
    print(costmatrix)
    # sliced_subcircuit_communication = dioc.sliced_subcircuit_communication_generate(remote_dag, slice_depth,
    #                                                                                 len(physical_bandwidth))
    #
    # max_value_test = np.sum(subcircuits_communication) * 2 / np.sum(physical_bandwidth)
    print(cost_matrix(dioc.normalize_subcircuit_communication(np.sum(subcircuits_communication) * 2 / np.sum(physical_bandwidth), subcircuits_communication), get_data_by_path('cfs')))


    # if allocation == 'trivial':
    #     subcircuits_allocation = dioc.trivial_allocate_subcircuit(len(subcircuit_qubit_partitions),
    #                                                               subcircuits_communication, srs_configurations)
    # elif allocation == 'random':
    #     subcircuits_allocation = dioc.random_allocate_subcircuit(len(subcircuit_qubit_partitions),
    #                                                              subcircuits_communication, srs_configurations)
    # elif allocation == 'greedy':
    #     subcircuits_allocation = dioc.greedy_allocate_subcircuit(len(subcircuit_qubit_partitions),
    #                                                              subcircuits_communication, get_data_by_path('cfs'))
    # # elif allocation == 'slice':
    # #     subcircuits_allocation = dioc.sliced_Hungarian_allocate_subcircuit(len(subcircuit_qubit_partitions),
    # #                                                                        normalized_sliced_subcircuits_communication,
    # #                                                                        get_data_by_path('cfs'))
    # else:
    #     subcircuits_allocation = dioc.Hungarian_allocate_subcircuit(len(subcircuit_qubit_partitions),
    #                                                                 normalized_subcircuits_communication,
    #                                                                 get_data_by_path('cfs'))
    # timestart = time.time()
    # if schedule == 'baseline':
    #     ecost, time_step, execution_schedule, discard_entanglement_count = dioc.time_evolution_old_only_remote_time_greedy(
    #         srs_configurations, circuit_dagtable,
    #         gate_list, qubit_loc_subcircuit_dic,
    #         subcircuits_allocation, remote_operations, swap_mode)
    # else:
    #     ecost, time_step, execution_schedule, discard_entanglement_count = dioc.time_evolution_greedy(
    #         srs_configurations, circuit_dagtable,
    #         gate_list, qubit_loc_subcircuit_dic,
    #         subcircuits_allocation, remote_operations,
    #         schedule, swap_mode)
    # process_time = time.time() - timestart
    ## bk
    normalized_subcircuits_communication = dioc.normalize_subcircuit_communication(max_value, subcircuits_communication)
    # costmatrix = cost_matrix(normalized_subcircuits_communication, get_data_by_path('srs'))
    sliced_subcircuits_communication = dioc.convolutional_sliced_subcircuit_communication_generate(
        remote_dag, slice_depth,
        len(physical_bandwidth), 0.1, 0.5, 0.4)
    # costmatrix = cost_matrix(normalized_subcircuits_communication, get_data_by_path('srs'))
    normalized_sliced_subcircuits_communication = dioc.normalize_slice_subcircuit_communication(
        sliced_subcircuits_communication, average_distance=2, physical_bandwidth=physical_bandwidth)
    slice_cost_matrix = cost_matrix_sliced(normalized_sliced_subcircuits_communication, get_data_by_path('cfs'))
    if allocation == 'trivial':
        subcircuits_allocation = dioc.trivial_allocate_subcircuit(len(subcircuit_qubit_partitions),
                                                                  subcircuits_communication, srs_configurations)
    elif allocation == 'random':
        subcircuits_allocation = dioc.random_allocate_subcircuit(len(subcircuit_qubit_partitions),
                                                                 subcircuits_communication, srs_configurations)
    elif allocation == 'greedy':
        subcircuits_allocation = dioc.greedy_allocate_subcircuit(len(subcircuit_qubit_partitions),
                                                                 subcircuits_communication, get_data_by_path('cfs'))
    elif allocation == 'slice':
        subcircuits_allocation = dioc.sliced_Hungarian_allocate_subcircuit(len(subcircuit_qubit_partitions),
                                                                           normalized_sliced_subcircuits_communication,
                                                                           get_data_by_path('cfs'))
    else:

        subcircuits_allocation = dioc.Hungarian_allocate_subcircuit(len(subcircuit_qubit_partitions),
                                                                    normalized_subcircuits_communication,
                                                                    get_data_by_path('cfs'))

    print(allocation)
    print(subcircuits_allocation)
    if allocation == 'hungarian':
        info_data = compute_mapping_information(costmatrix, subcircuits_allocation)
    elif allocation == 'slice':
        info_data = compute_mapping_information(slice_cost_matrix, subcircuits_allocation)
    else:
    # print(subcircuits_communication)
    # print(slice_cost_matrix)
    # print(np.sum(subcircuits_communication, axis=1))
    # print(np.sum(get_data_by_path('cfs'), axis=1))
    # print(subcircuits_allocation)
        info_data = compute_mapping_information(slice_cost_matrix, subcircuits_allocation)
    info_data['opt'] = max_value
    # print(info_data)
    info_data['mean'] = np.mean(sliced_subcircuits_communication)
    info_data['cost_mean'] = np.mean(slice_cost_matrix)
    if info_data['mean'] != 0:
        info_data['cost_var'] = np.var(slice_cost_matrix) / info_data['mean']
    else:
        info_data['cost_var'] = np.var(slice_cost_matrix)
    timestart = time.time()
    if schedule == 'baseline':
        ecost, time_step, execution_schedule, discard_entanglement_count = dioc.time_evolution_old_only_remote_time_greedy(
            srs_configurations, circuit_dagtable,
            gate_list, qubit_loc_subcircuit_dic,
            subcircuits_allocation, remote_operations, swap_mode)
    else:
        ecost, time_step, execution_schedule, discard_entanglement_count = dioc.time_evolution_greedy(
            srs_configurations, circuit_dagtable,
            gate_list, qubit_loc_subcircuit_dic,
            subcircuits_allocation, remote_operations,
            schedule, swap_mode)
    process_time = time.time() - timestart

    # df.loc[circuitname] = np.mean(np.array(data),axis=0)
    print(qasm_path)
    # print(info_data['zero_num'])
    print(process_time)
    return [time_step, ecost, process_time, discard_entanglement_count, info_data['cost'], info_data['opt'], info_data['cost_mean'], info_data['cost_var']]
    # return [time_step, ecost, process_time, discard_entanglement_count]


if __name__ == "__main__":
    import sys

    # w = sys.argv[1]
    # s = int(sys.argv[2])
    # SWAP_MODE = sys.argv[3]
    # name = 'h_global'
    # schedules = [0, 1]
    # schedule = 2
    # if w == 'baseline':
    #     schedule = 0
    small_device_qubit_number = 5
    large_device_qubit_number = 40
    qubit_per_channels = ['hetero_random']
    N_samples = 500

    # p_gen=1
    # p_swap=0.95
    # q_swap=0.12
    # p_cons=0.0
    # cutoff=10
    test_para = 'allocation'
    # to_test = ['hungarian', 'trivial', 'greedy', 'slice']  # 'hungarian', 'trivial',
    to_test = ['greedy', 'random']
    # random_qswap_0.12_qubit_per_channel_hetero_random_p_swap0.95_p_cons0.05_cutoff_10
    paras = {
        'p_gen': 1,
        'p_swap': 0.95,
        'q_swap': 0.12,
        'p_cons': 0.05,
        'cutoff': 10,
        'swap_mode': 'algebraic_connectivity',  # random algebraic_connectivity total_distance
        'channels': 'hetero_random',
        'allocation': 'hungarian',  #
        'schedule': 'global',  # global direct indirect baseline
        'samples': N_samples
    }

    # filepath = '/home/normaluser/fzchen/TQMR/haha/test/circuit_qasmbench_onlycx/large_h/multiplier_n75.qasm'
    # filepath = '/home/normaluser/fzchen/qnet_iwqos/qnet_iwqos/test_for_weight/rca_300.qasm'
    # filepath = '/home/normaluser/fzchen/qnet_iwqos/qnet_iwqos/test_for_weight/rca_200.qasm'
    # filepath = '/home/normaluser/fzchen/qnet_iwqos/qnet_iwqos/test_for_weight/qft_100.qasm'
    # filepath = '/home/normaluser/fzchen/qnet_iwqos/qnet_iwqos/pra_benchmark/vqe/vqe_real_amplitudes_100_reps_full1.qasm'
    # filepath = '/home/normaluser/fzchen/qnet_iwqos/qnet_iwqos/test_for_weight/qaoa_max_cut_50.qasm'
    # paras['channels'] = dioc.generate_physical_bandwidth_grid(9, 5)
    # print(paras['channels'])

    # circuit_execution(filepath, paras['p_gen'], paras['p_swap'], paras['q_swap'],
    #                   paras['p_cons'], paras['cutoff'], paras['swap_mode'], paras['allocation'],
    #                   paras['channels'], paras['schedule'], 5, 40)

    # count_time = 0
    # for i in range(N_samples):
    #     count_time += circuit_execution(filepath, paras['p_gen'], paras['p_swap'], paras['q_swap'],
    #                       paras['p_cons'], paras['cutoff'], paras['swap_mode'], paras['allocation'],
    #                       paras['channels'], paras['schedule'], 5, 40)[0]
    #     print(count_time)
    # print(count_time/N_samples)

    for bandwidth in qubit_per_channels:
        paras['channels'] = bandwidth
        tasks_list = [[] for _ in range(len(to_test))]

        # path = '/home/normaluser/fzchen/qnet_iwqos/qnet_iwqos/test_for_weight'
        # path = '/home/normaluser/fzchen/qnet_iwqos/qnet_iwqos/one_circuit'
        path = '/home/normaluser/fzchen/qnet_iwqos/qnet_iwqos/pra_benchmark/vqe'
        # path = '/home/normaluser/fzchen/TQMR/haha/test/circuit_qasmbench_onlycx/large_h'
        # path = '/home/normaluser/hflash/qnet_iwqos/test_data/test_circuit_qft'
        # path = '/home/normaluser/fzchen/qnet_iwqos/qnet_iwqos/pra_benchmark'
        # path = '/home/normaluser/hflash/qnet_iwqos/exp_circuit_benchmark/readable_qubits'
        tmp_data_dir = f'/home/normaluser/hflash/qnet_iwqos/test_data/test_data_new_normalization'
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.qasm'):
                    qasm_path = os.path.join(root, file)
                    circuitname = file.split('.qasm')[-2]
                    # paras['channels'] = dioc.generate_physical_bandwidth_grid(9, 5)
                    for i in range(len(to_test)):
                        paras[test_para] = to_test[i]
                        # print(paras['channels'])
                        tasks_list[i].append(Task(qasm_path, circuitname, paras, tmp_data_dir))

        run_parallel(tasks_list, small_device_qubit_number, large_device_qubit_number)

        for tasks in tasks_list:
            df = pd.DataFrame(
                columns=['time cost', 'entanglement cost', 'processing time', 'discard entanglements', 'cost', 'opt', 'cost_mean', 'cost_var'])
            for task in tasks:
                df.loc[task.circuit_name] = np.min(np.array(task.get_data()), axis=0)
                # df.loc[task.circuit_name] = np.mean(np.array(task.get_data()), axis=0)
            save_dir = f'/home/normaluser/hflash/qnet_iwqos/test_data/alg_test/allocation'
            os.makedirs(save_dir, exist_ok=True)
            log_name = "{}_{}_{}_{}_{}".format(tasks[0].swap_mode, tasks[0].allocation, tasks[0].schedule,
                                               'hetero_random', getattr(tasks[0], test_para))
            # log_name = "{}_{}_{}_{}_{}".format(tasks[0].swap_mode, tasks[0].allocation, tasks[0].schedule,
            #                                    tasks[0].channels, getattr(tasks[0], test_para))
            # with open(f'{save_dir}/{log_name}','w') as f:
            #     for index, row in df.iterrows():
            #         f.write(f'{index}\n{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\n')
            df.to_csv(f'{save_dir}/{log_name}.csv', index=True)
