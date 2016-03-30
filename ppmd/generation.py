import host
import access
import data
import build

def get_type_map_symbol():
    return '_TYPE_MAP'

def get_first_index_symbol():
    return '_i'

def get_second_index_symbol():
    return '_j'

def get_first_cell_is_halo_symbol():
    return '_cp_halo_flag'

def get_second_cell_is_halo_symbol():
    return '_cpp_halo_flag'

_nl = '\n'

def generate_map(pair=True, symbol_external=None, symbol_internal=None, dat=None, access_type=None):
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

    _c = build.Code()
    _c += '#undef ' + symbol_internal + _nl

    if type(dat) is data.TypedDat:
        #case for typed dat
        _ncomp = dat.ncol

        # Define the entry point into the map
        _c += '#define ' + symbol_internal + '(x,y) ' + symbol_internal + '_##x(y)' + _nl

        # define the first particle map
        _c += '#define ' + symbol_internal + '_0(y) ' + symbol_external + \
              '[LINIDX_2D(' + str(_ncomp) + ',' + get_first_index_symbol() + \
              ',' + '(y)]' + _nl
        # second particle map
        _c += '#define ' + symbol_internal + '_1(y) ' + symbol_external + \
              '[LINIDX_2D(' + str(_ncomp) + ',' + get_second_index_symbol() + \
              ',' + '(y)]' + _nl

        return _c

    # case when dat is an ScalarArray
    elif issubclass(type(dat), host.Array):

        # If the ScalarArray is "halo aware" then when one or more of the
        # particles are in the halo the pointer should point to the second
        # element. In a reduction case this redirection should be at the
        # reduction stage of the kernel execution

        _c = build.Code()
        _c += '#undef ' + symbol_internal + _nl

        if (dat.halo_aware is True) and pair is True:

            # if condition is true we are in a halo, else we are in the
            # interior
            # TODO: Make the offset correct for ScalarArrays that are more than
            # TODO: one element.

            _c += '#define ' + symbol_internal + '(x) ' + \
                  '(((_cp_halo_flag || _cpp_halo_flag) > 0 ) ? ( ' + \
                  symbol_external + '[(x) + ' + str(1) + ']) : (' + \
                  symbol_external + '[(x)]' \
                  '))' + _nl

        else:
            # case where we map straight through to the external dat.
            _c += '#define ' + symbol_internal + '(x) ' + symbol_external + \
                  '[(x)]' + _nl

        return _c

    elif issubclass(type(dat), host.Matrix):
        # Case for host.Matrix and data.ParticleDat.

        _c = build.Code()
        _c += '#undef ' + symbol_internal + _nl

        _ncomp = str(dat.ncol)

        if pair:
            _c += '#define ' + symbol_internal + '(x,y) ' + symbol_internal + '_##x(y)' + _nl
            _c += '#define ' + symbol_internal + '_0(y) (' + symbol_external + \
                  '[' + get_first_index_symbol() + '*' + _ncomp + '+ (y) ])' + _nl

            _c += '#define ' + symbol_internal + '_1(y) (' + symbol_external + \
                  '[' + get_second_index_symbol() + '*' + _ncomp + '+ (y) ])' + _nl

        else:
            _c += '#define ' + symbol_internal + '(y) (' + symbol_external + \
                  '[' + get_first_index_symbol() + '*' + _ncomp + ' + (y)])' + \
                  _nl


        return _c

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
                _s = _space + host.ctypes_map[dat.dtype] + symbol_internal + '[' + str(dat.ncomp) + '] = { 0 };\n'

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

    return ''






