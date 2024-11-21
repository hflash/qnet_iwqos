import re
import logging
import itertools
import warnings


class Register(object):
    """Implement a generic register."""

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = 'reg'

    def __init__(self, size, name=None):
        """Create a new generic register.

        .. deprecated:: 0.5
            The `name` parameter will be optional in upcoming versions (>0.5.0)
            and the order of the parameters will change (`size`, `name`)
            instead of (`name`, `size`).
        """

        if isinstance(size, str):
            warnings.warn(
                "name will be optional in upcoming versions (>0.5.0) "
                "and order will be size, name.", DeprecationWarning)
            name_temp = size
            size = name
            name = name_temp

        if name is None:
            name = '%s%i' % (self.prefix, next(self.instances_counter))

        if not isinstance(name, str):
            print("The circuit name should be a string "
                  "(or None for autogenerate a name).")

        self.name = name
        self.size = size
        if size <= 0:
            print("register size must be positive")

    def __str__(self):
        """Return a string representing the register."""
        return "Register(%s,%d)" % (self.name, self.size)

    def __len__(self):
        """Return register size"""
        return self.size


class QuantumRegister(Register):
    """Implement a quantum register."""
    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = 'q'

    def qasm(self):
        """Return OPENQASM string for this register."""
        return "qreg %s[%d];" % (self.name, self.size)

    def __str__(self):
        """Return a string representing the register."""
        return "QuantumRegister(%s,%d)" % (self.name, self.size)

    def __len__(self):
        """Return a int representing the size."""
        return self.size

    def initialize(self):
        pass


class ClassicalRegister(Register):
    """Implement a classical register."""

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = 'c'

    def qasm(self):
        """Return OPENQASM string for this register."""
        return "creg %s[%d];" % (self.name, self.size)

    def __str__(self):
        """Return a string representing the register."""
        return "ClassicalRegister(%s,%d)" % (self.name, self.size)

    def __len__(self):
        """Return a int representing the size."""
        return self.size

    def initialize(self):
        pass
