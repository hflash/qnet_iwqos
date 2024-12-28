class Gate:
    def __init__(self, id, q1_loc, q2_loc, q1, q2):
        self.id = id
        self.q1_loc = q1_loc
        self.q2_loc = q2_loc
        self.q1_front: Gate = None
        self.q2_front: Gate = None
        self.successors: list[Gate] = []
        self.successors_cnt = 0
        self.has_executed = False

    def get_successors_cnt(self):
        return self.successors_cnt


class RemoteDag:
    def __init__(self, qubit_cnt, remote_operations, circuit_gate_list, qubit_loc_subcircuit_dic):
        self.gate_list: list[Gate] = []
        self.qubit_last_gate: list[Gate] = [None for _ in range(qubit_cnt)]
        self.gate_dict = {}
        for gate_id in remote_operations:
            gate = circuit_gate_list[gate_id]
            qubit1 = gate.get_qubits()[0]
            qubit2 = gate.get_qubits()[1]
            qubit1_loc = qubit_loc_subcircuit_dic[qubit1]
            qubit2_loc = qubit_loc_subcircuit_dic[qubit2]

            self.add_Gate(gate_id, qubit1_loc, qubit2_loc, qubit1, qubit2)
        self.calculate_cnt()

    def add_Gate(self, id, q1_loc, q2_loc, q1, q2):
        tmp_Gate = Gate(id, q1_loc, q2_loc, q1, q2)
        tmp_Gate.q1_front = self.qubit_last_gate[q1]
        tmp_Gate.q2_front = self.qubit_last_gate[q2]
        if self.qubit_last_gate[q1]:
            self.qubit_last_gate[q1].successors.append(tmp_Gate)
        if self.qubit_last_gate[q2]:
            self.qubit_last_gate[q2].successors.append(tmp_Gate)
        self.qubit_last_gate[q1] = tmp_Gate
        self.qubit_last_gate[q2] = tmp_Gate
        self.gate_list.append(tmp_Gate)
        self.gate_dict[id] = tmp_Gate

    def calculate_cnt(self):
        from collections import deque
        in_degree = {gate: (gate.q1_front != None) + (gate.q2_front != None) for gate in self.gate_list}

        # 拓扑排序
        topo_order = []
        queue = deque(gate for gate in self.gate_list if in_degree[gate] == 0)
        while queue:
            current = queue.popleft()
            topo_order.append(current)
            for succ in current.successors:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        # 动态规划计算后继节点数量
        successor_count = {gate: 0 for gate in self.gate_list}  # 每个节点的后继节点数量
        for gate in reversed(topo_order):  # 逆拓扑顺序
            for succ in gate.successors:
                gate.successors_cnt += succ.successors_cnt + 1  # 累加后继节点数量

        for gate in self.gate_list:
            gate.successors_cnt = successor_count[gate]

    def get_front_layer(self):
        front_layer = []
        for gate in self.gate_list:
            if (gate.q1_front is None or gate.q1_front.has_executed) and (
                    gate.q2_front is None or gate.q2_front.has_executed) and not gate.has_executed:
                front_layer.append(gate.id)
        return front_layer

    def execute_gate(self, gateid: int):
        if gateid in self.gate_dict:
            self.gate_dict[gateid].has_executed = True