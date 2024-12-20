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


class Task:
    def __init__(self,circuit_path, name, sample_num,bandwidth, save_path,test_value):
        self.lock = Lock()
        self.circuit_path = circuit_path
        self.circuit_name = name
        self.sample_num = sample_num
        self.bandwidth = bandwidth
        self.test_value = test_value
        self.data = []
        self.has_done = 0
        self.data_path = os.path.join(save_path, '{}_{}_testv_{:.2f}'.format(name, bandwidth, test_value))
    def call_back(self, result):
        with self.lock:
            self.data.append(result)
            self.has_done += 1
            if self.has_done == self.sample_num:
                self.dump_data()

    def dump_data(self):
        with open(self.data_path, 'w') as f:
            f.write('time cost \t\t\t entanglement \t\t\t costprocessing time \t\t\t discard entanglements\n')
            for row in self.data:
                f.write('{:<9} \t\t\t {:<12} \t\t\t {:<19} \t\t\t {}\n'.format(row[0], row[1], row[2], row[3]))


    def get_data(self):
        return self.data

def run_parallel(tasks_list:list[list[Task]], schedule, qswap, cutoff, small_device_qubit_number, large_device_qubit_number,w):

    p = Pool()
    for tasks in tasks_list:
        for task in tasks:
            for _ in range(task.sample_num):
                p.apply_async(circuit_execution,
                              (task.circuit_path, schedule, qswap, cutoff, task.bandwidth, small_device_qubit_number, large_device_qubit_number,w,task.test_value),
                              callback=task.call_back)
                                    
    p.close()
    p.join()


# L1: 线路分割+线路映射
# L2: 线路执行优化
def circuit_execution(qasm_path, schedule, qswap, cutoff, qubit_per_channel, small_device_qubit_number, large_device_qubit_number,w,test_value):
    if 'small' in qasm_path:
        scale = 'small'
        device_qubit_number = small_device_qubit_number
    else:
        scale = 'large'
        device_qubit_number = large_device_qubit_number
    randomseed = np.random.seed()
    # randomseed = 0
    timestart = time.time()
    remote_operations, circuit_dagtable, gate_list, subcircuits_communication, qubit_loc_subcircuit_dic, subcircuit_qubit_partitions = circuitPartition(
        qasm_path, device_qubit_number, randomseed)
    srs_configurations = dioc.srs_config_chain(qubit_per_channel=qubit_per_channel, p_gen = 1, p_swap= 0.95,  q_swap=qswap, p_cons = test_value, cutoff=cutoff, randomseed=randomseed)
    subcircuits_allocation = dioc.Hungarian_allocate_subcircuit(len(subcircuit_qubit_partitions),
                                                                subcircuits_communication,
                                                                srs_configurations['adj'])

    timestart = time.time()
    if w == 'baseline':
        ecost, time_step, execution_schedule, discard_entanglement_count = dioc.time_evolution_old_only_remote_time_greedy(
            srs_configurations, circuit_dagtable,
            gate_list, qubit_loc_subcircuit_dic,
            subcircuits_allocation, remote_operations,
            schedule)
    else:
        ecost, time_step, execution_schedule, discard_entanglement_count = dioc.time_evolution_greedy(
        srs_configurations, circuit_dagtable,
        gate_list, qubit_loc_subcircuit_dic,
        subcircuits_allocation, remote_operations,
        schedule,w,test_value)
    process_time = time.time() - timestart

    #df.loc[circuitname] = np.mean(np.array(data),axis=0)
    print(qasm_path)
    return [time_step, ecost, process_time, discard_entanglement_count]


if __name__ == "__main__":
    import sys
    w = sys.argv[1]
    s = int(sys.argv[2])
    # name = 'h_global'
    # schedules = [0, 1]
    schedule = 2
    small_device_qubit_number = 5
    large_device_qubit_number = 13
    qubit_per_channel = 1
    N_samples = 20

    to_test = [i*0.05 for i in range(20)]

    p_gen=1
    p_swap=0.95
    q_swap=0.12
    p_cons=0.25
    cutoff=10

    tasks_list = [[] for _ in range(len(to_test))]

    path = '/home/normaluser/fzchen/qnet_iwqos/qnet_iwqos/1220_test/circuits'
    tmp_data_dir = '/home/normaluser/fzchen/qnet_iwqos/qnet_iwqos/1220_test/result/'+w
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.qasm'):
                qasm_path = os.path.join(root, file)
                circuitname = file.split('.qasm')[-2]
                for i in range(len(to_test)):
                    tasks_list[i].append(Task(qasm_path, circuitname, N_samples, qubit_per_channel, tmp_data_dir, to_test[i]))

    

    run_parallel(tasks_list, schedule, q_swap, cutoff, small_device_qubit_number, large_device_qubit_number, w)

    
    for tasks in tasks_list:
        df = pd.DataFrame(columns=['time cost', 'entanglement cost', 'processing time', 'discard entanglements'])
        for task in tasks:
            df.loc[task.circuit_name] = np.mean(np.array(task.get_data()),axis=0)
        with open('test_data/{}/test_greedy_{}_{}_{:.2f}_log'.format(w,w,qubit_per_channel,tasks[0].test_value),'w') as f:
            for index, row in df.iterrows():
                f.write(f'{index}\n{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\n')

