"""
Create the C++ classes to access A.i[k] where A.i[k+1] is adjacent in memory
to A.i[k] using structs not classes
"""

from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

from cgen import *

class DSLStructComp(object):
    def __init__(self, sym, i_gather_sym, j_gather_sym, ctype, const,
            ncomp, i_index, j_index, stride=None):
        """
        Creates the C/C++ required to access A.i[x] and A.j[x].

        :param sym: symbol used in kernel e.g. 'P'.
        :param ctype: c datatype to use e.g. 'double'.
        :param const: bool, True/False to indicate write mode is read only.
        :param ncomp: number of components per particle in ParticleDat.
        :param i_index: symbol used as first loop index e.g. '_i'.
        :param j_index: symbol used as second loop index e.g. '_j'.
        """

        c = 'const' if const else ''
        self.header = Line(
            """
            struct _{sym}_call {{
                {const} {ctype} * RESTRICT i;
                {const} {ctype} * RESTRICT j;
            }};
            """.format(
                ctype=ctype,
                const=c,
                sym=sym
            )
        )
        """Returns the class definitions."""

        t = "_{sym}_call".format(sym=sym)
        v = sym
        self.kernel_arg_decl = Value(t, v)
        """Returns the parameter decleration for the kernel"""
        
        self.kernel_create_arg = Line("""
       _{sym}_call _{sym}_c = {{ {i_gather_sym}+{i_index}*{ncomp}, {j_gather_sym}+{j_index}*{ncomp} }};
        """.format(
            sym=sym,
            i_gather_sym=i_gather_sym,
            j_gather_sym=j_gather_sym,
            i_index=i_index,
            j_index=j_index,
            ncomp=ncomp
        ))
        """Returns the construction of the argument to call the kernel"""
        
        self.kernel_arg = "_{sym}_c".format(sym=sym)
    
    def __repr__(self):
        return str(self.header) + "\n"+ str(self.kernel_arg_decl) + \
                str(self.kernel_create_arg) + "\n" + \
                str(self.kernel_arg)



