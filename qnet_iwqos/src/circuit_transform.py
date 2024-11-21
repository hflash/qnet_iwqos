from qiskit import QuantumCircuit, transpile
import os
import logging
from qiskit.providers.fake_provider.utils.configurable_backend import *

if __name__ == '__main__':
    transform_log_path = "../logs/circuit_transform_log_large.txt"
    coupling_map_all = []
    qubit_num_small = 16
    for i in range(qubit_num_small):
        for j in range(qubit_num_small):
            if i != j:
                coupling_map_all.append([i, j])
    backend_small = ConfigurableFakeBackend(version=1, name='circuit_tranform', n_qubits=qubit_num_small, coupling_map=coupling_map_all)
    coupling_map_all = []
    qubit_num_medium = 27
    for i in range(qubit_num_medium):
        for j in range(qubit_num_medium):
            if i != j:
                coupling_map_all.append([i, j])
    backend_medium = ConfigurableFakeBackend(version=1, name='circuit_tranform', n_qubits=qubit_num_medium,
                                            coupling_map=coupling_map_all)
    coupling_map_all = []
    qubit_num_large = 433
    for i in range(qubit_num_large):
        for j in range(qubit_num_large):
            if i != j:
                coupling_map_all.append([i, j])
    backend_large = ConfigurableFakeBackend(version=1, name='circuit_tranform', n_qubits=qubit_num_large,
                                             coupling_map=coupling_map_all)
    read_path = '../../QASMBench-master/large'
    write_path = '../../circuit_qasmbench_onlycx'
    for root, dirs, files in os.walk(read_path, topdown=False):
        for name in files:
            if 'qasm' in name and 'transpiled' not in name and 'cc' not in name:
                file_path = os.path.join(root, name)
                print(file_path)
                try:
                    circuit = QuantumCircuit.from_qasm_file(file_path)
                    circuit_len = circuit.depth()
                    if len(circuit.qubits) <= 16:
                        if 'small' in file_path:
                            circuit_transpile = transpile(circuit, layout_method='trivial', backend=backend_small, optimization_level=0)
                            path_write = os.path.join(write_path + "/small", name)
                            with open(path_write, 'w') as f:
                                f.write(circuit_transpile.qasm())
                            with open(transform_log_path, 'a') as f:
                                if circuit_len == circuit_transpile.depth():
                                    f.write(file_path + "transform succeed!\n")
                                else:
                                    f.write(file_path + str(circuit_len) + " : " + str(circuit_transpile.depth()) + " transform not succeed with depth inconsistency!\n")
                    if 16 < len(circuit.qubits) <= 27:
                        if 'medium' in file_path:
                            circuit_transpile = transpile(circuit, layout_method='trivial', backend=backend_medium,
                                                          optimization_level=0)
                            path_write = os.path.join(write_path + "/medium", name)
                            with open(path_write, 'w') as f:
                                f.write(circuit_transpile.qasm())
                            with open(transform_log_path, 'a') as f:
                                if circuit_len == circuit_transpile.depth():
                                    f.write(file_path + "transform succeed!\n")
                                    # print("succeed")
                                else:
                                    f.write(file_path + str(circuit_len) + " : " + str(circuit_transpile.depth()) + " transform not succeed with depth inconsistency!\n")
                                    # print("failed")
                    if len(circuit.qubits) > 27:
                        if 'large' in file_path:
                            circuit_transpile = transpile(circuit, layout_method='trivial', backend=backend_large,
                                                          optimization_level=0)
                            path_write = os.path.join(write_path + "/large", name)
                            with open(path_write, 'w') as f:
                                f.write(circuit_transpile.qasm())
                            with open(transform_log_path, 'a') as f:
                                if circuit_len == circuit_transpile.depth():
                                    f.write(file_path + "transform succeed!\n")
                                else:
                                    f.write(file_path + str(circuit_len) + " : " + str(circuit_transpile.depth()) +" transform not succeed with depth inconsistency!\n")

                except Exception as e:
                    print(e)
                    continue

