import host
import access
import data

def get_type_map_symbol():
    return '_TYPE_MAP'

def get_first_index_symbol():
    return '_i'

def get_second_index_symbol():
    return '_j'


def generate_map(pair=True, symbol_external=None, symbol_internal=None, dat=None, access_type=None, n3_disable_dats=[]):
    """
    Create pointer map.

    :param pair:
    :param symbol_external:
    :param symbol_internal:
    :param dat:
    :param access_type:
    :param n3_disable_dats:
    :return:
    """

    assert symbol_external is not None, "generate_map error: No symbol_external"
    assert symbol_internal is not None, "generate_map error: No symbol_internal"
    assert dat is not None, "generate_map error: No dat"
    assert access_type is not None, "generate_map error: No access_type"

    _space = ' ' * 14

    if pair:
        _n = 2
    else:
        _n = 1


    if type(dat) is data.TypedDat:
        #case for typed dat
        _s = '\n'
        _s += _space + host.ctypes_map[dat.dtype] + ' *' + symbol_internal + '[' + str(_n) +'];\n'
        _s += _space + symbol_internal + '[0] = &' + symbol_external + '[' + str(dat.ncol) + '* _TYPE_MAP[_i]];\n'
        if pair:
            _s += _space + symbol_internal + '[1] = &' + symbol_external + '[' + str(dat.ncol) + '* _TYPE_MAP[_j]];\n'

        return _s

    elif issubclass(type(dat), host.Array):
        return '\n'

    elif issubclass(type(dat), host.Matrix):
        # Case for host.Matrix and data.ParticleDat.

        if dat in n3_disable_dats:
            # Point the second element at a null_array that should be removed by compiler optimiser.
            _ncomp = dat.ncol
            _s = '\n'
            _s += _space + host.ctypes_map[dat.dtype] + ' *' + symbol_internal + '[' + str(_n) +'];\n'
            _s += _space + symbol_internal + '[0] = ' + symbol_external + '+' + str(_ncomp) + '* _i;\n'
            if pair:
                _s += _space + host.ctypes_map[dat.dtype] + ' _null_' + symbol_internal + '[' + str(_ncomp) + '] = {0}; \n'
                _s += _space + symbol_internal + '[1] = ' + '&_null_' + symbol_internal + '[0];\n'

        else:
            _ncomp = dat.ncol
            _s = '\n'
            _s += _space + host.ctypes_map[dat.dtype] + ' *' + symbol_internal + '[' + str(_n) +'];\n'
            _s += _space + symbol_internal + '[0] = ' + symbol_external + '+' + str(_ncomp) + '* _i;\n'
            if pair:
                _s += _space + symbol_internal + '[1] = ' + symbol_external + '+' + str(_ncomp) + '* _j;\n'

        return _s

    else:

        raise Exception("Generate mapping failure: unknown dat type.")



def generate_reduction_init_stage(symbol_external, symbol_internal, dat, access_type):
    """
    Create the code to initialise the code for an INC, INC0 access descriptor for x86.
    :arg string symbol_external: variable name for shared library
    :arg string symbol_internal: variable name for kernel.
    :arg data dat: :class:`~data.ParticleDat` or :class:`~data.ScalarArray` data object in question.
    :arg access access_type: Access being used.

    :return: string for initialisation code.
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
                _s = _space + host.ctypes_map[dat.dtype] + ' ' + symbol_internal + '[' + str(dat.ncomp) + '] = { 0 };\n'

            else:

                raise Exception("Generate mapping failure: Incremental access_type not found.")

        return _s + '\n'
    else:
        return ''



def generate_reduction_pre_kernel_stage(symbol_external, symbol_internal, dat, access_type):
    """
    Create the code to handle reductions in the pre kernel part of the code.
    :arg string symbol_external: variable name for shared library
    :arg string symbol_internal: variable name for kernel.
    :arg data dat: :class:`~data.ParticleDat` or :class:`~data.ScalarArray` data object in question.
    :arg access access_type: Access being used.

    :return: string for initialisation code.
    """

    return ''


def generate_reduction_kernel_stage(symbol_external, symbol_internal, dat, access_type):
    """
    Create the code to handle reductions in the looping part of the code.
    :arg string symbol_external: variable name for shared library
    :arg string symbol_internal: variable name for kernel.
    :arg data dat: :class:`~data.ParticleDat` or :class:`~data.ScalarArray` data object in question.
    :arg access access_type: Access being used.

    :return: string for initialisation code.
    """

    return ''



def generate_reduction_final_stage(symbol_external, symbol_internal, dat, access_type):
    """
    Create the code to finalise the reduction.
    :arg string symbol_external: variable name for shared library
    :arg string symbol_internal: variable name for kernel.
    :arg data dat: :class:`~data.ParticleDat` or :class:`~data.ScalarArray` data object in question.
    :arg access access_type: Access being used.

    :return: string for initialisation code.
    """

    _space = ' ' * 14
    if issubclass(type(dat), host.Array):
        # Case for host.Array and data.ScalarArray.
        if not access_type.incremented:
            return ''

        else:

            _s = _space + '#pragma omp critical \n' + _space + '{ \n'

            _s += 2 * _space + 'for(int _rix = 0; _rix < ' + str(dat.ncomp) + '; _rix++){ \n'

            _s += 3 * _space + symbol_external + '[_rix] += ' + symbol_internal + '[_rix]; \n'

            _s += 2 * _space + '} \n'

            _s += _space + '} \n'


        return _s + '\n'
    else:
        return ''






