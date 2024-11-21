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
# @Time     : 2024/11/5 10:34
# @Author   : HFLASH @ LINKE
# @File     : fakeprvovider_trial.py
# @Software : PyCharm

from qiskit.providers.fake_provider import GenericBackendV2
from qiskit import transpile

coupling_map_all = []
num_qubits = 100
for i in range(num_qubits):
    for j in range(num_qubits):
        if i != j:
            coupling_map_all.append([i, j])
backend = GenericBackendV2(basis_gates=["x", "y", "z", "h", "s", "t", "cx", "rx", "rz", "ry", "tdg"],
                                num_qubits=num_qubits, coupling_map=coupling_map_all)
print(backend.coupling_map)
# circuit_transpile = transpile(qft_circuit, layout_method='trivial', backend=backend)
# bankend = GenericBackendV2