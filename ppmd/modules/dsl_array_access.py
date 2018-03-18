"""
Generate code to access arrays
"""

from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

from cgen import *

class DSLArrayAccess(object):
    def __init__(self, sym, ctype, const, ncomp):
        
        kernel_arg = Pointer(
            Value(
                ctype,
                'RESTRICT ' + sym
        )) 

        if const:
            kernel_arg = Const(kernel_arg)

        self.kernel_arg_decl = kernel_arg
        """argument decleration in kernel func"""
        

        self.kernel_create_i_arg = ''
        self.kernel_create_j_arg = Line('')
        """gather code pre loop"""

        self.kernel_create_i_scatter = ''
        """scatter code post loop"""


        self.kernel_arg = sym
        """symbol to call with"""

class DSLGlobalArrayAccess(object):
    def __init__(self, sym, thread_sym, ctype, const, ncomp):
        
        kernel_arg = Pointer(
            Value(
                ctype,
                'RESTRICT ' + sym
        )) 

        if const:
            kernel_arg = Const(kernel_arg)

        self.kernel_arg_decl = kernel_arg
        """argument decleration in kernel func"""
        
        self.kernel_create_j_arg = Line('')
        """gather code pre loop"""

        call_sym = thread_sym if const else sym+'_red'

        self.kernel_arg = call_sym
        """symbol to call with"""
        
        iscatter = ''
        igather = ''
        
        if not const:
            igather += """
            {ctype} {call_sym}[{ncomp}] = {{ 0 }};
            """.format(
                ctype=ctype,
                call_sym=call_sym,
                ncomp=ncomp
            )
            iscatter += """
            for(INT64 {li}=0 ; {li}<{ncomp} ; {li}++){{
                {thread_sym}[{li}] += {call_sym}[{li}];
            }}
            """.format(
                call_sym=call_sym,
                ncomp=ncomp,
                thread_sym=thread_sym,
                li='_'+thread_sym+sym
            )


        self.kernel_create_i_arg = igather
        """the gather"""

        self.kernel_create_i_scatter = iscatter
        """scatter code post loop"""


