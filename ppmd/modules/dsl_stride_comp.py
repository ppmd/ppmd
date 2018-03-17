"""
Create the C++ classes to access A.i[k] where A.i[k+1] is adjacent in memory
to A.i[k];
"""

from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

from cgen import *

class DSLStrideComp(object):
    def __init__(self, sym, i_gather_sym, j_gather_sym, ctype, const, ncomp, 
            i_index, j_index, stride):
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
            class _{sym}_class {{
               public:
                _{sym}_class(
                    {ctype} {const} * RESTRICT ptr,
                    const INT64 stride
                ) {{
                    pptr=ptr;
                    pstride=stride;
                }};
                _{sym}_class() {{return;}};
                {const} {ctype}& operator[] (INT64 ind) {{return pptr[ind*pstride];}};
              private:               
                {ctype} {const} * RESTRICT pptr;
                INT64 pstride; 
            }};
            class _{sym}_call {{
              public:
                _{sym}_class i;
                _{sym}_class j;
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
       _{sym}_call _{sym}_c;
       _{sym}_c.i = _{sym}_class({i_gather_sym}+{i_index}, {stride});
       _{sym}_c.j = _{sym}_class({j_gather_sym}+{j_index}, {stride});
        """.format(
            sym=sym,
            i_gather_sym=i_gather_sym,
            j_gather_sym=j_gather_sym,
            i_index=i_index,
            j_index=j_index,
            stride=stride
        ))
        """Returns the construction of the argument to call the kernel"""
        
        self.kernel_arg = "_{sym}_c".format(sym=sym)
    
    def __repr__(self):
        return str(self.header) + "\n"+ str(self.kernel_arg_decl) + \
                str(self.kernel_create_arg) + "\n" + \
                str(self.kernel_arg)



