import sympy as sp

def is_symbol_input(obj):
    return isinstance(obj, sp.Symbol)

class Gate:
    def __init__(self, name):
        self.name = name
        self.is_symbol = False
        self.gateset = {"X", "Y", "Z", "H", "S", "T", "CX", "RX", "RZ", "TDG"}

    def get_name(self):
        return self.name

    def get_qubits(self):
        pass  # We'll define this in the subclasses


class X(Gate):


    def __init__(self, qubit):
        name = "X"
        super().__init__(name)
        self.qubit = qubit

    def get_qubits(self):
        return [self.qubit]


class Y(Gate):
    def __init__(self, qubit):
        name = "Y"
        super().__init__(name)
        self.qubit = qubit

    def get_qubits(self):
        return [self.qubit]


class Z(Gate):
    def __init__(self, qubit):
        name = "Z"
        super().__init__(name)
        self.qubit = qubit

    def get_qubits(self):
        return [self.qubit]


class H(Gate):
    def __init__(self, qubit):
        name = "H"
        super().__init__(name)
        self.qubit = qubit

    def get_qubits(self):
        return [self.qubit]


class T(Gate):
    def __init__(self, qubit):
        name = "T"
        super().__init__(name)
        self.qubit = qubit

    def get_qubits(self):
        return [self.qubit]

class TDG(Gate):
    def __init__(self, qubit):
        name = "TDG"
        super().__init__(name)
        self.qubit = qubit

    def get_qubits(self):
        return [self.qubit]


class S(Gate):
    def __init__(self, qubit):
        name = "S"
        super().__init__(name)
        self.qubit = qubit

    def get_qubits(self):
        return [self.qubit]


class RX(Gate):
    def __init__(self, qubit, para):
        name = "RX"
        super().__init__(name)
        self.qubit = qubit
        self.para = para
        if is_symbol_input(para):
            self.is_symbol = True
        else:
            self.is_symbol = False


    def get_qubits(self):
        return [self.qubit]

    def get_para(self):
        return self.para

    def bind_para(self,para1):
        assert is_symbol_input(para1) == False
        return RX(self.qubit,para1)


class RZ(Gate):
    def __init__(self, qubit, para):
        name = "RZ"
        super().__init__(name)
        self.qubit = qubit
        self.para = para
        if is_symbol_input(para):
            self.is_symbol = True
        else:
            self.is_symbol = False

    def get_qubits(self):
        return [self.qubit]

    def get_para(self):
        return self.para

    def bind_para(self,para1):
        assert is_symbol_input(para1) == False
        return RZ(self.qubit,para1)


class RY(Gate):
    def __init__(self, qubit, para):
        name = "RY"
        super().__init__(name)
        self.qubit = qubit
        self.para = para
        if is_symbol_input(para):
            self.is_symbol = True
        else:
            self.is_symbol = False

    def get_qubits(self):
        return [self.qubit]

    def get_para(self):
        return self.para

    def bind_para(self,para1):
        assert is_symbol_input(para1) == False
        return RY(self.qubit,para1)

class P(Gate):
    def __init__(self, qubit, para):
        name = "P"
        super().__init__(name)
        self.qubit = qubit
        self.para = para
        if is_symbol_input(para):
            self.is_symbol = True
        else:
            self.is_symbol = False

    def get_qubits(self):
        return [self.qubit]

    def get_para(self):
        return self.para

    def bind_para(self,para1):
        assert is_symbol_input(para1) == False
        return P(self.qubit,para1)

class CP(Gate):
    def __init__(self, control_qubit, target_qubit,para):
        name = "CP"
        super().__init__(name)
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit
        self.para = para
        if is_symbol_input(para):
            self.is_symbol = True
        else:
            self.is_symbol = False

    def get_qubits(self):
        '''
        :return: [control_qubit,target_qubit]
        '''
        return [self.control_qubit,self.target_qubit]

    def get_para(self):
        return self.para

    def bind_para(self,para1):
        assert is_symbol_input(para1) == False
        return CP(self.control_qubit,self.target_qubit,para1)

class CX(Gate):
    def __init__(self, control_qubit, target_qubit):
        name = "CX"
        super().__init__(name)
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit

    def get_qubits(self):
        '''
        :return: [control_qubit,target_qubit]
        '''
        return [self.control_qubit, self.target_qubit]

class SWAP(Gate):
    def __init__(self, qubit1, qubit2):
        name = "SWAP"
        super().__init__(name)
        self.qubit1 = qubit1
        self.qubit2 = qubit2

    def get_qubits(self):
        '''
        :return: [control_qubit,target_qubit]
        '''
        return [self.qubit1, self.qubit2]


class CRY(Gate):
    def __init__(self, control_qubit, target_qubit,para):
        name = "CRY"
        super().__init__(name)
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit
        self.para = para
        if is_symbol_input(para):
            self.is_symbol = True
        else:
            self.is_symbol = False

    def get_qubits(self):
        '''
        :return: [control_qubit,target_qubit]
        '''
        return [self.control_qubit,self.target_qubit]

    def get_para(self):
        return self.para

    def bind_para(self,para1):
        assert is_symbol_input(para1) == False
        return CRY(self.control_qubit,self.target_qubit,para1)

class Init(Gate):
    def __init__(self, qubit):
        name = "Init"
        super().__init__(name)
        self.qubit = list(range(qubit))

    def get_qubits(self):
        return self.qubit


if __name__ == "__main__":
    x = sp.symbols('x')
    rx1 = RX(2,x)
    print(rx1.is_symbol)
    print(rx1.para)
    rx2=rx1.bind_para(0.4)
    print("====================")
    print(rx1.is_symbol)
    print(rx1.para)
    print(rx2.is_symbol)
    print(rx2.para)
