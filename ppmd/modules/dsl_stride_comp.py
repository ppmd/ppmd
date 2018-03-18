"""
Create the C++ classes to access A.i[k] where A.i[k+1] is offset in memory
to A.i[k] by a non-unit stride;
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

        self.isymbol = "__{sym}i".format(sym=sym)
        self.jsymbol = "__{sym}j".format(sym=sym)
    
        c = 'const'

        header = """
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
                {const} {ctype}& operator[] (INT64 ind) 
                    {{return pptr[ind*pstride];}};
              private:               
                {ctype} {const} * RESTRICT pptr;
                INT64 pstride; 
            }};
            """.format( ctype=ctype, const=c, sym=sym)

        #if const:
        #    header += """
        #    class _{sym}_call {{
        #      public:
        #        _{sym}_class i;
        #        _{sym}_class j;
        #    }};
        #    """.format( ctype=ctype, sym=sym) 

        #else:
        #    header += """
        #    class _{sym}_call {{
        #      public:
        #        {ctype} * RESTRICT i;
        #        _{sym}_class j;
        #    }};
        #    """.format( ctype=ctype, sym=sym) 


        self.header = Line(header)
        """Returns the class definitions."""
        
        jt = "_{sym}_class".format(sym=sym)
        if const:
            it = jt
        else:
            it = "{ctype} * RESTRICT".format(ctype=ctype)

        self.kernel_arg_decl = (Value(it, self.isymbol), Value(jt, self.jsymbol))

        """Returns the parameter decleration for the kernel"""
        
        iarg = """
        //_{sym}_class _{sym}_ci;
        """.format(sym=sym)
        
        iscatter = """
        """

        if const:
            iarg += """
            _{sym}_class _{sym}_ci({i_gather_sym}+{i_index}, {stride});
            """.format(
                sym=sym,
                i_gather_sym=i_gather_sym,
                j_gather_sym=j_gather_sym,
                i_index=i_index,
                j_index=j_index,
                stride=stride
            )
            iscatter += ''
        else:
            iarg += """
            {ctype} _{sym}_ci[{ncomp}] = {{0}};
            for( INT64 _{sym}ix=0 ; _{sym}ix<{ncomp} ; _{sym}ix++ ){{
                _{sym}_ci[_{sym}ix] = {i_gather_sym}[{i_index} + _{sym}ix * {stride}];
            }}
            """.format(
                sym=sym,
                ctype=ctype,
                i_gather_sym=i_gather_sym,
                j_gather_sym=j_gather_sym,
                i_index=i_index,
                j_index=j_index,
                stride=stride,
                ncomp=ncomp
            )
            iscatter += """
            for( INT64 _{sym}ix=0 ; _{sym}ix<{ncomp} ; _{sym}ix++ ){{
                 {i_gather_sym}[{i_index} + _{sym}ix * {stride}] = _{sym}_ci[_{sym}ix];
            }}
            """.format(
                sym=sym,
                ctype=ctype,
                i_gather_sym=i_gather_sym,
                j_gather_sym=j_gather_sym,
                i_index=i_index,
                j_index=j_index,
                stride=stride,
                ncomp=ncomp,
            )


        self.kernel_create_i_arg = iarg
        """Returns the construction of the argument to call the kernel"""
        
        self.kernel_create_i_scatter = iscatter
        """Returns the construction of the argument to call the kernel"""
 



        self.kernel_create_j_arg = Line("""
        _{sym}_class _{sym}_cj({j_gather_sym}+{j_index}, {stride});
        """.format(
            sym=sym,
            i_gather_sym=i_gather_sym,
            j_gather_sym=j_gather_sym,
            i_index=i_index,
            j_index=j_index,
            stride=stride
        ))
        """Returns the construction of the argument to call the kernel"""
        
        self.kernel_arg = "_{sym}_ci, _{sym}_cj".format(sym=sym)


    def __repr__(self):
        return str(self.header) + "\n"+ str(self.kernel_arg_decl) + \
                str(self.kernel_create_arg) + "\n" + \
                str(self.kernel_arg)



