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
# @Time     : 2024/2/18 11:00
# @Author   : HFLASH @ LINKE
# @File     : generate_benchmark.py
# @Software : PyCharm

from qiskit import transpile
from qiskit.circuit.library import RealAmplitudes
# from qiskit.providers.fake_provider import ConfigurableFakeBackend
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.qasm2 import dumps
from qiskit import QuantumCircuit
import numpy as np
from qiskit.circuit.library import QFT
from qiskit.circuit.library import CDKMRippleCarryAdder
from qiskit.circuit.library import GroverOperator
import re
from math import pi
from src.quantumcircuit.QAOAcircuit import QAOAcircuit
# from qiskit import Aer
# from qiskit import Aer
# from qiskit.algorithms.optimizers import COBYLA
import networkx as nx
import numpy as np


# def vqe_circuit_generation(num_qubits, rep):
#     ansatz = RealAmplitudes(num_qubits, entanglement='full', reps=rep)
#     # ansatz = RealAmplitudes(num_qubits, reps=rep)
#     params = np.random.random(len(ansatz.parameters))
#     circuit = ansatz.bind_parameters(params)
#     # print(ansatz)
#     coupling_map_all = []
#     qubit_num = num_qubits
#     path_write = '../circuit_benchmark/vqe/vqe_real_amplitudes_' + str(num_qubits) + '_reps_full' + str(rep) + '.qasm'
#     for i in range(qubit_num):
#         for j in range(qubit_num):
#             if i != j:
#                 coupling_map_all.append([i, j])
#     backend = ConfigurableFakeBackend(version=1,
#                                       single_qubit_gates=["x", "y", "z", "h", "s", "t", "rx", "rz", "ry", "tdg"],
#                                       basis_gates=["x", "y", "z", "h", "s", "t", "cx", "rx", "rz", "ry", "tdg"],
#                                       name='circuit_tranform', n_qubits=qubit_num, coupling_map=coupling_map_all)
#
#     circuit_transpile = transpile(circuit, layout_method='trivial', backend=backend)
#     with open(path_write, 'w') as f:
#         f.write(circuit_transpile.qasm())
#     # with open(path_write, 'r+') as f:
#     #     data = []
#     #     for line in f.readlines():
#     #         if 'qreg' in line:
#     #             assert ('qreg q[' + str(qubit_num) + '];' in line)
#     #             line = 'qreg q[%d];\n' % (num_qubits)
#     #             data.append(line)
#     #             print(line)
#     #         elif 'measure' in line:
#     #             print(line)
#     #             line = ""
#     #             data.append(line)
#     #             print(data)
#     #         else:
#     #             data.append(line)
#     #     print(data)
#     #     with open(path_write, 'w') as f:
#     #         for item in data:
#     #             f.write(item)
#     print(circuit_transpile.depth())
#     # print(circuit_transpile.)
def count_cx(circuit):
    cxcount = 0
    for gate in circuit.data:
        if gate.operation.name == 'cx':
            cxcount += 1
    return cxcount

def generate_qaoa_circuit(num_qubits):
    path_write = '../circuit_benchmark/qaoa/qaoa_' + str(num_qubits) + '.qasm'

    G = nx.erdos_renyi_graph(num_qubits, 0.15)
    # nx.draw(G, with_labels=True)
    # plt.show()
    # print(G.edges)
    num_qubits = G.number_of_nodes()
    print(num_qubits)
    qaoa_qc = QAOAcircuit(num_qubits=num_qubits, graph=G)
    circuit = qaoa_qc.constructed_circuit(None)
    qasm = circuit.to_QASM(path_write)
    print(qasm)


def create_qft_circuit(num_qubits):
    # 创建一个QFT电路
    qft_circuit = QFT(num_qubits)
    # print(qft_circuit.qasm())
    # print(qft_circuit.depth())
    path_write = '../exp_circuit_benchmark/pra_benchmark/qft/qft_' + str(num_qubits) + '.qasm'
    # write_qasm = qft_circuit.qasm()
    # with open(path_write, 'w') as f:
    #     f.write(write_qasm)
    #     print("Write succeed!")
    coupling_map_all = []
    # num_qubits = 100
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i != j:
                coupling_map_all.append([i, j])
    backend = GenericBackendV2(basis_gates=["x", "y", "z", "h", "s", "t", "cx", "rx", "rz", "ry", "tdg"],
                               num_qubits=num_qubits, coupling_map=coupling_map_all)
    circuit_transpile = transpile(qft_circuit, layout_method='trivial', backend=backend)
    print(dumps(circuit_transpile))
    print(circuit_transpile.depth())
    write_qasm = dumps(circuit_transpile)
    print(count_cx(circuit_transpile))
    def replace_pi(match):
        expression = match.group(0)  # 获取匹配到的整个表达式
        # print(eval(expression))
        return "%.5f" %(eval(expression))
        # if expression == 'pi':  # 如果表达式仅为pi
        #     return str(pi)
        # else:  # 如果表达式形如pi/n
        #     divisor = float(expression.split('/')[1])  # 获取n的值
        #     return "%.5f" %(pi / divisor)  # 计算pi/n的值并转换为字符串

    # 使用正则表达式查找所有pi或pi/n形式的表达式并替换它们
    replaced_qasm = re.sub(r'((-?\d+\*)?pi(/\d+)?)', replace_pi, write_qasm)
    replaced_string = replaced_qasm.replace("; ", ";\n")
    with open(path_write, 'w') as f:
        f.write(replaced_string)
        print("Write succeed!")
    return qft_circuit


# def create_large_rca(num_qubits):
#     path_write = '../circuit_benchmark/rca/rca_' + str(num_qubits) + '.qasm'
#     # 检查量子比特数量是否符合纹波进位加法器的要求
#     if num_qubits % 4 != 0:
#         raise ValueError("Number of qubits should be a multiple of 4 for a ripple carry adder.")
#
#     # 创建纹波进位加法器电路
#     rca_circuit = CDKMRippleCarryAdder(num_state_qubits=num_qubits // 2 - 1, name="large_rca")
#     print(rca_circuit.qasm())
#     print(rca_circuit.depth())
#     coupling_map_all = []
#     for i in range(num_qubits):
#         for j in range(num_qubits):
#             if i != j:
#                 coupling_map_all.append([i, j])
#     backend = ConfigurableFakeBackend(version=1,
#                                       single_qubit_gates=["x", "y", "z", "h", "s", "t", "rx", "rz", "ry", "tdg"],
#                                       basis_gates=["x", "y", "z", "h", "s", "t", "cx", "rx", "rz", "ry", "tdg"],
#                                       name='circuit_tranform', n_qubits=num_qubits, coupling_map=coupling_map_all)
#
#     circuit_transpile = transpile(rca_circuit, layout_method='trivial', backend=backend)
#     print(circuit_transpile.qasm())
#     print(circuit_transpile.depth())
#     with open(path_write, 'w') as f:
#         f.write(circuit_transpile.qasm())
#         print("Write succeed!")
#     return rca_circuit


# def create_grover_circuit(num_qubits):
#     path_write = '../circuit_benchmark/grover/grover_' + str(num_qubits) + '.qasm'
#     # 创建随机oracle
#     oracle = QuantumCircuit(num_qubits)
#     random_angles = np.random.uniform(low=0, high=2*np.pi, size=num_qubits)
#     for qubit in range(num_qubits):
#         oracle.x(qubit)
#     # oracle.z(num_qubits-1)  # 假设最后一个量子比特处于“好”的状态
#
#     # 创建随机状态准备电路
#     # state_preparation = QuantumCircuit(num_qubits)
#     # random_angles = np.random.uniform(low=0, high=2*np.pi, size=num_qubits)
#     # for qubit in range(num_qubits):
#     #     state_preparation.ry(random_angles[qubit], qubit)
#
#     # 创建Grover算法的操作符
#     grover_op = GroverOperator(oracle)
#     grover_circuit = grover_op.decompose()
#
#     # print(grover_circuit.qasm())
#     # print(grover_circuit.depth())
#     coupling_map_all = []
#     for i in range(num_qubits):
#         for j in range(num_qubits):
#             if i != j:
#                 coupling_map_all.append([i, j])
#     backend = ConfigurableFakeBackend(version=1,
#                                       single_qubit_gates=["x", "y", "z", "h", "s", "t", "rx", "rz", "ry", "tdg"],
#                                       basis_gates=["x", "y", "z", "h", "s", "t", "cx", "rx", "rz", "ry", "tdg"],
#                                       name='circuit_tranform', n_qubits=num_qubits, coupling_map=coupling_map_all)
#
#     circuit_transpile = transpile(grover_circuit, layout_method='trivial', backend=backend)
#     # print(circuit_transpile.qasm())
#     # print(circuit_transpile.depth())
#     write_qasm = circuit_transpile.qasm()
#
#     def replace_pi(match):
#         expression = match.group(0)  # 获取匹配到的整个表达式
#         if expression == 'pi':  # 如果表达式仅为pi
#             return str(pi)
#         else:  # 如果表达式形如pi/n
#             divisor = float(expression.split('/')[1])  # 获取n的值
#             return str(pi / divisor)  # 计算pi/n的值并转换为字符串
#
#     # 使用正则表达式查找所有pi或pi/n形式的表达式并替换它们
#     replaced_qasm = re.sub(r'pi(/\d+)?', replace_pi, write_qasm)
#     with open(path_write, 'w') as f:
#         f.write(replaced_qasm)
#         print("Write succeed!")
#     return grover_circuit


if __name__ == '__main__':
    # vqe_circuit_generation(30, 1)
    # vqe_circuit_generation(60, 1)
    num_qubits_list = [100, 200, 300]  # 量子比特数列表
    # num_qubits_list = [100]
    for num_qubits in num_qubits_list:
    # #     create_large_rca(num_qubits)
        circuit = create_qft_circuit(num_qubits)
        # print(circuit.qasm())
    # num_qubits = 5  # 更改此值以构建不同大小的电路
    # qaoa_circuit = generate_qaoa_circuit(num_qubits, 1)
    #
    # # 打印电路信息
    # print("Number of qubits in the circuit:", len(qaoa_circuit[0].qubits))
    # 创建QAOA量子线路
    # circuit = create_qaoa_circuit(100, p=1)

    # 打印量子线路的一部分信息
    # print(f"Circuit depth: {circuit[0].depth()}")
    # 尝试创建大规模的纹波进位加法器电路
    # num_qubits_list = [100, 200, 300]  # 量子比特数列表
    # for num_qubits in num_qubits_list:
    #     try:
    #         rca_circuit = create_large_rca(num_qubits)
    #         print(f"Successfully created a RippleCarryAdder circuit with {num_qubits} qubits.")
    #     except Exception as e:
    #         print(f"Failed to create a RippleCarryAdder circuit with {num_qubits} qubits. Error: {e}")

    # 选择量子比特的数量和QAOA层数
    # num_qubits = 100  # 可以改为200或300
    # p = 100  # QAOA的深度
    #
    # # 创建QAOA线路
    # qaoa_circuit = create_qaoa_circuit(num_qubits, p)
    #
    # # 打印线路信息（由于量子比特数量可能很大，此行可能输出大量信息）
    #
    #
    # print(qaoa_circuit.qasm())

    # num_qubits_list = [100, 200, 300]
    # 示例：创建一个含有x量子比特的Grover电路
    # for num_qubits in num_qubits_list:
    #     grover_circuit = generate_qaoa_circuit(num_qubits)
    #     分解并打印电路
    #     print()

    pass

