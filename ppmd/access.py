"""
This module contains the access descriptor class and the pre-defined access descriptors to use when passing
instances of ParticleDat and ScalarArray to the build system.
"""

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

_lookup = {'R': 'Read only',
           'W': 'Write only',
           'RW': 'Read and write',
           'INC': 'Incremental',
           'NULL': 'NULL'
           }


class AccessType(object):
    """
    Class to hold an access descriptor for data. In a pyop2 style manner. (WIP)

    :arg str mode: Access mode, must be from: ``"R", "W", "RW", "INC"``
    """
    _modes = ["R", "W", "RW", "INC", "NULL"]

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
        return self._mode in ["W", "RW", "INC"]

R = AccessType("R")
"""Access descriptor for read only data. """

W = AccessType("W")
"""Access descriptor for write only data. """

RW = AccessType("RW")
"""Descriptor for data that has accessed for both read and write. """

INC = AccessType("INC")
"""Access descriptor for data that is incremented. """

NULL = AccessType("NULL")
"""NULL access descriptor for data. """

def generate_reduction_init_stage(symbol_external, symbol_internal, dat, access_type):
    """
    Create the code to initialise the code for an INC, INC0 access descriptor for x86.
    :arg string symbol_external: variable name for shared library
    :arg string symbol_internal: variable name for kernel.
    :arg data dat: :class:`~data.ParticleDat` or :class:`~data.ScalarArray` data object in question.
    :arg access access_type: Access being used.

    :return: string for initialisation code.
    """








































