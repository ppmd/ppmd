import access
import data
import host

def get_first_index_symbol():
    return '_ix'

def create_local_reduction_vars_arrays(symbol_external, symbol_internal, dat, access_type):
    """
    This should go before the if statement on the thread id in the kernel.
    :param symbol_external:
    :param symbol_internal:
    :param dat:
    :param access_type:
    :return:
    """

    _space = ' ' * 14
    if issubclass(type(dat), host.Array):
        # Case for host.Array and data.ScalarArray.
        if not access_type.incremented:
            _s = _space + host.ctypes_map[dat.dtype] + ' *' + symbol_internal + ' = ' + symbol_external + ';\n'

        else:
            if access_type is access.INC:
                _s = _space + host.ctypes_map[dat.dtype] + ' ' + symbol_internal + '[' + str(dat.ncomp) + '];\n'
                _s += _space + 'memcpy( &'+ symbol_internal + '[0],' + '&'+ symbol_external + '[0],' + str(dat.ncomp) + ' * sizeof(' + host.ctypes_map[dat.dtype] + ')) ;\n'

            elif access_type is access.INC0:
                _s = _space + host.ctypes_map[dat.dtype] + symbol_internal + '[' + str(dat.ncomp) + '] = { 0 };\n'

            else:

                raise Exception("Generate mapping failure: Incremental access_type not found.")

        return _s + '\n'
    else:
        return ''


def create_pre_loop_map_matrices(pair=True, symbol_external=None, symbol_internal=None, dat=None, access_type=None):
    """
    This should go after the if statement in the pair loop and is associated with particle i.

    :param symbol_external:
    :param symbol_internal:
    :param dat:
    :param access_type:
    :return:
    """

    assert symbol_external is not None, "create_pre_loop_map_matrices error: No symbol_external"
    assert symbol_internal is not None, "create_pre_loop_map_matrices error: No symbol_internal"
    assert dat is not None, "create_pre_loop_map_matrices error: No dat"
    assert access_type is not None, "generate_mapgenerate_map error: No access_type"

    _space = ' ' * 14
    if issubclass(type(dat), host.Matrix):
        # Case for host.Array and data.ScalarArray.
        if pair is True:
            _s = _space + host.ctypes_map[dat.dtype] + ' *' + symbol_internal + '[2]; \n'
        else:
            _s = _space + host.ctypes_map[dat.dtype] + ' *' + symbol_internal + '; \n'

        if not access_type.incremented:

            # First pointer, the one which is associated with the thread.
            if pair is True:
                _s += _space + symbol_internal + '[0] = ' + symbol_external + '+ _ix *' + str(dat.ncomp) + ' ;\n'
            else:
                _s += _space + symbol_internal + ' = ' + symbol_external + '+ _ix *' + str(dat.ncomp) + ';\n'

        else:
            # Create a local copy/new row vector for the row in the particle dat.
            if access_type is access.INC:
                _s = _space + host.ctypes_map[dat.dtype] + ' _tmp_' + symbol_internal + '[' + str(dat.ncomp) + '];\n'

                _s += _space + 'memcpy( &_tmp_'+ symbol_internal + '[0],' + '&'+ symbol_external + '[' + '+ _ix *' + str(dat.ncomp) + '],' + str(dat.ncomp) + ' * sizeof(' + host.ctypes_map[dat.dtype] + ')) ;\n'

            elif access_type is access.INC0:
                _s = _space + host.ctypes_map[dat.dtype] + ' _tmp_' + symbol_internal + '[' + str(dat.ncomp) + '] = { 0 };\n'

            else:
                raise Exception("Generate mapping failure: Incremental access_type not found.")

            # assign first pointer to hopefully local copy here.
            if pair is True:
                _s += _space + symbol_internal + '[0] = _tmp_' + symbol_internal + ' ;\n'
            else:
                _s += _space + symbol_internal + ' = _tmp_' + symbol_internal + ' ;\n'


        return _s + '\n'
    else:
        return ''




