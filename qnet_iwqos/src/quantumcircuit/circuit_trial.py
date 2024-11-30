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
# @Time    : 2024/2/12 09:52
# @Author  : HFALSH @ LINKE
# @File    : circuit_trial.py
# @IDE     : PyCharm

from src.quantumcircuit.circuit import QuantumCircuit
from src.quantumcircuit.gate import *

if __name__ == "__main__":
    # quantum_circuit =QuantumCircuit.random_circuit(num_qubits=5, depth=5,gate1p=0.1,gate2p=0.2)
    # quantum_circuit.to_QASM("random_test.qasm")
    # quantum_circuit=QuantumCircuit.from_QASM("random_test.qasm")
    # quantum_circuit.print_dag_circuit()

    # quantum_circuit = QuantumCircuit.load_alg("block")
    # quantum_circuit.print_dag_circuit()
    # print(quantum_circuit.to_QASM_list())

    # qc = QuantumCircuit.from_QASM("random_test.qasm")
    # print(qc.to_QASM("readfile_test.qasm"))
    quantum_circuit = QuantumCircuit(3)
    quantum_circuit.add_gate(X(1))
    quantum_circuit.add_gate(CX(1,2))
    quantum_circuit.add_gate(CX(0, 1))
    quantum_circuit.add_gate(H(0))
    quantum_circuit1 = QuantumCircuit(3)
    quantum_circuit1.add_gate(X(1))
    quantum_circuit1.add_gate(CX(1,2))
    quantum_circuit1.add_gate(CX(0, 1))
    quantum_circuit1.add_gate(H(0))
    # quantum_circuit.add_QuantumCircuit(quantum_circuit1)
    print(quantum_circuit.to_dagtable())
    # print(quantum_circuit.qubit_number)
    # print(quantum_circuit.gate_set)
    # a=quantum_circuit.get_qubit_duration_path(0)
    # for path in a:
    #     print("========")
    #     for pp in path:
    #         print(pp.get_name())
    # a = quantum_circuit.get_qubit_duration_path(1)
    # print("\n")
    # for path in a:
    #     print("========")
    #     for pp in path:
    #         print(pp.get_name())
    # a = quantum_circuit.get_qubit_duration_path(2)
    # print("\n")
    # for path in a:
    #     print("========")
    #     for pp in path:
    #         print(pp.get_name())
    # print(quantum_circuit.gate_number)
    # quantum_circuit.to_QASM("aaaa.qasm")
