"""
Create the C/C++ to load the required data from all particles in a cell
into temporary arrays for cell by cell pairlooping.
"""

from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import ctypes
from cgen import *



def DSLRecordLocal(ind_sym, nlocal_sym, store_sym, store_ind_sym, count_sym):
    
    init = Line('INT64 {count_sym} = 0;'.format(count_sym=count_sym))
    loop_block = Line("""
    if ({ind_sym} < {nlocal_sym} ){{
        {count_sym} += 1;
    }}
    {store_sym}[{store_ind_sym}] = {ind_sym};
    """.format(
            ind_sym=ind_sym,
            nlocal_sym=nlocal_sym,
            store_sym=store_sym,
            store_ind_sym=store_ind_sym,
            count_sym=count_sym
        )
    )

    return init, loop_block


