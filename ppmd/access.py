"""
This module contains the access descriptor class and the pre-defined access descriptors to use when passing
instances of ParticleDat and ScalarArray to the build system.
"""
from __future__ import division, print_function
import ctypes

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


"""
rst_doc{

.. contents:

access Module
=============

.. automodule:: access

Access Type Class
~~~~~~~~~~~~~~~~~

.. autoclass:: access.AccessType
   :show-inheritance:
   :undoc-members:
   :members:

Predefined Access Modes
~~~~~~~~~~~~~~~~~~~~~~~

These should be used when passing instances of :class:`~data.ParticleDat` or
:class:`~data.ScalarArray` to the build system to declare the access mode required
by the associated kernel.

.. autodata:: RW
.. autodata:: R
.. autodata:: W
.. autodata:: INC

}rst_doc
"""



_ctypes_map = {ctypes.c_double: 'double',
              ctypes.c_int: 'int',
              ctypes.c_long: 'long',
              ctypes.c_byte: 'char',
              'float64': 'double',
              'int32': 'int',
              'doublepointerpointer': 'double **',
              # ctypes.c_longlong: 'long long',
              'doublepointer': 'double *',
              'intpointer': 'int *',
              'longpointer': 'long *'}




_lookup = {'R': 'Read only',
           'W': 'Write only',
           'RW': 'Read and write',
           'INC': 'Incremental',
           'INC0': 'Incremental from zero',
           'NULL': 'NULL'
           }


class AccessType(object):
    """
    Class to hold an access descriptor for data. In a pyop2 style manner. (WIP)

    :arg str mode: Access mode, must be from: ``"R", "W", "RW", "INC"``
    """
    _modes = ["R", "W", "RW", "INC", "NULL","INC0"]

    def __init__(self, mode):


        self._mode = mode

    def __str__(self):
        return self._mode

    def __repr__(self):

        return "%(MODE)s." % {'MODE':_lookup[self._mode]}

    @property
    def mode(self):
        """
        :return: The held access mode.
        """
        return self._mode

    @property
    def read(self):
        """
        Does this access type read the data, True/False.
        :return: Bool.
        """
        return self._mode in ["R", "RW", "INC"]

    @property
    def write(self):
        """
        Does this access type write data, True/False.
        :return: Bool.
        """
        return self._mode in ["W", "RW", "INC", "INC0"]

    @property
    def incremented(self):
        """
        Does this access perform reductions, True/False.
        :return: Bool
        """
        return self._mode in ["INC", "INC0"]


R = AccessType("R")
W = AccessType("W")
INC0 = AccessType("INC0")

RW = AccessType("RW")
"""Descriptor for data that has accessed for both read and write. """

INC = AccessType("INC")
"""Access descriptor for data that is incremented. """

NULL = AccessType("NULL")
"""NULL access descriptor for data. """

READ = R
"""Access descriptor for read only data. """
WRITE = W
"""Access descriptor for write only data. """
INC_ZERO = INC0
"""Access descriptor for data that is incremented from zero. """


all_access_types = (READ, WRITE, INC_ZERO, INC, RW)


class DatArgStore(object):
    def __init__(self, allow, initial):
        """
        Provide compatibility checking of passed initial dats with allowed dats
        and provide consistent looping over dats.
        :param allow: Dictionary, form {dat-type: tuple of allowed access descriptors)}
        :param initial: initial dat dict
        """
        assert type(allow) is dict, "expected a dict"
        assert type(initial) is dict, "expected a dict"
        self.allow = allow
        self._check_args_allowed(initial)
        self.initial = initial
        self.register = {}
        self.symbols = dict()
        self.looping_tuples = ()
        self.dats = tuple()
        self._register_initial()


    def _register_initial(self):
        """Hash the properties of the initial args to allow checking of alternate
        args, plus determine an order for args for consistent looping"""

        dats = []
        objs = []
        for ix, ax in enumerate(self.initial.items()):
            symbol = ax[0]
            obj = ax[1][0]
            mode = ax[1][1]
            # define the order for new dat dicts

            assert obj not in objs, "dats may not be passed twice"
            objs.append(obj)

            self.symbols[symbol] = ix
            self.register[symbol] = (
                mode,
                type(obj),
                obj.dtype,
                obj.ncomp
            )
            dats.append(ax)

        self.dats = tuple(dats)

    def _check_args_allowed(self, args):
        """Check args are in the set of allowed args and access descriptors"""
        for ax in args.items():
            assert len(ax) == 2, "error in passed dat, missing symbol or dat?"
            assert len(ax[1]) > 1, "error in passed dat, missing access descriptor?" + str(ax[1])

            symbol = ax[0]
            obj = type(ax[1][0])
            mode = ax[1][1]
            assert obj in self.allow.keys(), "Passed Dat is not a compatible type: " + str(obj)
            assert mode in self.allow[obj], "Passed access descriptor is not compatible: " + \
                                            str(mode)
            assert type(symbol) is str, "Passed symbol is not a str: " + str(symbol)

    def _check_new_dats(self, new_dats):
        assert len(new_dats.items()) == len(self.dats), "incorrect number of dats"
        assert type(new_dats) is dict, "expected a dictonary of new dats"
        objs = []
        for ax in new_dats.items():
            assert len(ax) == 2, "error in passed dat, missing symbol or dat?"
            assert len(ax[1]) > 1, "error in passed dat, missing access descriptor?"

            symbol = ax[0]
            obj = ax[1][0]
            mode = ax[1][1]

            assert obj not in objs, "dats may not be passed twice"
            objs.append(obj)

            assert symbol in self.symbols.keys(), "unexpected symbol in dat dict"
            assert mode == self.register[symbol][0], "incorrect access mode"
            assert issubclass(type(obj), self.register[symbol][1]), "incompatible dat passed"+\
                str(type(obj)) + " != " + str(self.register[symbol][1])
            assert obj.dtype == self.register[symbol][2], "incompatible dat data type"
            assert obj.ncomp == self.register[symbol][3], "incompatible dat ncomp"

    def items(self, new_dats=None):
        """return the dats in a guaranteed order"""
        if new_dats is None:
            return self.dats

        # check new dats are valid
        self._check_new_dats(new_dats)

        # ensure order is consistent with old dats
        dats = [0] * len(self.symbols)
        for ax in new_dats.items():
            dats[self.symbols[ax[0]]] = ax

        return tuple(dats)



class StaticArgStore(object):
    def __init__(self, initial):
        """
        Provide compatibility checking of passed initial dats with allowed dats
        and provide consistent looping over dats.
        :param allow: Dictionary, form {dat-type: tuple of allowed access descriptors)}
        :param initial: initial dat dict
        """
        assert type(initial) is dict, "expected a dict"
        self.allow = _ctypes_map.keys()
        self._check_args_allowed(initial)
        self.initial = initial
        self.register = {}
        self.symbols = dict()
        self.looping_tuples = ()
        self.dats = tuple()
        self._register_initial()

    def _register_initial(self):
        """Hash the properties of the initial args to allow checking of alternate
        args, plus determine an order for args for consistent looping"""

        dats = []
        objs = []
        for ix, ax in enumerate(self.initial.items()):
            obj = ax[0]
            ctype = ax[1]
            # define the order for new dat dicts

            assert obj not in objs, "dats may not be passed twice"
            objs.append(obj)
            self.symbols[obj] = ix
            self.register[obj] = ctype
            dats.append(ax)

        self.dats = tuple(dats)

    def _check_args_allowed(self, args):
        """Check args are in the set of allowed args and access descriptors"""
        for ax in args.items():
            assert len(ax) == 2, "error in passed dat, missing symbol"
            symbol = ax[0]
            ctype = ax[1]
            assert ctype in self.allow, "Passed arg is not a compatible type: " + str(type)
            assert type(symbol) is str, "Passed symbol is not a str: " + str(symbol)

    def _check_new_dats(self, new_dats):
        assert len(new_dats.items()) == len(self.dats), "incorrect number of dats"
        assert type(new_dats) is dict, "expected a dictonary of new dats"
        objs = []
        for ax in new_dats.items():
            assert len(ax) == 2, "error in passed dict, missing symbol or dat?"
            symbol = ax[0]
            assert symbol not in objs, "dats may not be passed twice"
            objs.append(symbol)
            assert symbol in self.symbols.keys(), "unexpected symbol in dict"

    def items(self):
        """return the dats in a guaranteed order"""
        return self.dats

    def get_args(self, values):
        # check new dats are valid
        self._check_new_dats(values)

        # ensure order is consistent with old dats
        dats = [0] * len(self.symbols)
        for ax in values.items():
            dats[self.symbols[ax[0]]] = self.register[ax[0]](ax[1])

        return dats














