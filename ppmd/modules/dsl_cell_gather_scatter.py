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

from ppmd.data import ParticleDat
from ppmd.host import ctypes_map

class DSLPartitionTempSpace(object):
    """
    Partition temporary space for cell by cell gather/scatter
    """
    def __init__(self, dat_dict, max_ncell, start_ptr, extras=()):
        
        self.idict = {}
        self.jdict = {}
        
        self.req_bytes = 0
        self._ptr_init = []


        for i, dat in enumerate(dat_dict.items()):

            obj = dat[1][0]
            mode = dat[1][1]
            symbol = dat[0]

            if issubclass(type(obj), ParticleDat):
                block_size = obj.ncomp * ctypes.sizeof(obj.dtype)
                ctype = ctypes_map[obj.dtype]

                self._ptr_init.append(
                    '{ctype}* RESTRICT _itmp_{sym} = ({ctype}* RESTRICT) \
( (void * RESTRICT) ( {src} + {max_ncell}*{offset} ));'.format(
                            ctype=ctype,
                            sym=symbol,
                            src=start_ptr,
                            max_ncell=max_ncell,
                            offset=self.req_bytes
                        )
                )
                self._ptr_init.append(
                    '{ctype}* RESTRICT _jtmp_{sym} = ({ctype}* RESTRICT) \
( (void * RESTRICT) ( {src} + {max_ncell}*{offset} ));'.format(
                            ctype=ctype,
                            sym=symbol,
                            src=start_ptr,
                            max_ncell=max_ncell,
                            offset=self.req_bytes + block_size
                        )
                )
                
                self.req_bytes += 2 * block_size
                self.idict[symbol] = '_itmp_{sym}'.format(sym=symbol)
                self.jdict[symbol] = '_jtmp_{sym}'.format(sym=symbol)

        for ex in extras:
            
            symbol = ex[0]
            ncomp = ex[1]
            dtype = ex[2]
            block_size = ncomp * ctypes.sizeof(dtype)
            ctype = ctypes_map[dtype]

            self._ptr_init.append(
                '{ctype}* RESTRICT _itmp_{sym} = ({ctype}* RESTRICT) \
( (void * RESTRICT) ( {src} + {max_ncell}*{offset} ));'.format(
                        ctype=ctype,
                        sym=symbol,
                        src=start_ptr,
                        max_ncell=max_ncell,
                        offset=self.req_bytes
                    )
            )
            self._ptr_init.append(
                '{ctype}* RESTRICT _jtmp_{sym} = ({ctype}* RESTRICT) \
( (void * RESTRICT) ( {src} + {max_ncell}*{offset} ));'.format(
                        ctype=ctype,
                        sym=symbol,
                        src=start_ptr,
                        max_ncell=max_ncell,
                        offset=self.req_bytes + block_size
                    )
            )
            
            self.req_bytes += 2 * block_size
            self.idict[symbol] = '_itmp_{sym}'.format(sym=symbol)
            self.jdict[symbol] = '_jtmp_{sym}'.format(sym=symbol)


        
        self._ptr_init_str = '\n'.join(self._ptr_init)
        self.ptr_init = Line(self._ptr_init_str)


def DSLSeqGather(src_sym, dst_sym, ncomp, src_ind, dst_ind):
    tmp_sym = '__'+src_sym+dst_sym
    
    b = Block(
        (
            Line(
                '{dst_sym}[{dst_ind}] = {src_sym}[{src_ind}];'.format(
                    dst_sym=dst_sym,
                    dst_ind=dst_ind+'*'+str(ncomp)+'+'+tmp_sym,
                    src_sym=src_sym,
                    src_ind=src_ind+'*'+str(ncomp)+'+'+tmp_sym
                )
            ),
        )
    )

    f0 = For(
        'INT64 {}=0'.format(tmp_sym),
        '{}<{}'.format(tmp_sym, ncomp),
        '{}++'.format(tmp_sym),
        b
    )
    return f0





