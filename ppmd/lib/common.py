from __future__ import print_function, division, absolute_import

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level
import cgen
import ctypes


allowed_dtypes = (
    ctypes.c_double,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int32,
    ctypes.c_uint32,
    ctypes.c_int64,
    ctypes.c_uint64,
    ctypes.c_byte,
    'float64',
    'int32'
)


_ctypes_map = {
    ctypes.c_int64: 'int64_t',
    ctypes.c_uint64: 'uint64_t',
    ctypes.c_int32: 'int32_t',
    ctypes.c_uint32: 'uint32_t'   
}


class DtypeToCtype:
    def __init__(self, existing):
        self._e = existing

    def __getitem__(self, key):
        if key in self._e.keys():
            return self._e[key]
        else:
            return cgen.dtype_to_ctype(key)


ctypes_map = DtypeToCtype(_ctypes_map)


