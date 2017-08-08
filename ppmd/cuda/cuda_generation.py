from __future__ import print_function, division, absolute_import
import ppmd.access as access
import ppmd.host as host

from ppmd.cuda import cuda_data, cuda_base, cuda_build

_nl = '\n'

def get_first_index_symbol():
    return '_ix'

def get_type_map_symbol():
    return '_TYPE_MAP'

def get_second_index_symbol():
    return '_iy'

def get_variable_prefix(access_mode):
    if access_mode is access.R:
        return 'const '
    else:
        return ' '

#####################################################################################
# block of code class
#####################################################################################

class Code(object):
    def __init__(self, init=''):
        self._c = str(init)

    @property
    def string(self):
        return self._c

    def add_line(self, line=''):
        self._c += '\n' + str(line)

    def add(self, code=''):
        self._c += str(code)

    def __iadd__(self, other):
        self.add(code=str(other))
        return self

    def __str__(self):
        return str(self._c)

    def __add__(self, other):
        return Code(self.string + str(other))


def create_host_function_argument_decleration(symbol=None, dat=None, mode=None, cc=None):

    assert symbol is not None, "No symbol passed"
    assert dat is not None, "No dat passed"
    assert mode is not None, "No mode"
    assert cc is not None, "No compiler"

    if not mode.write:
        const_str = 'const '
    else:
        const_str = ''

    return const_str + host.ctypes_map[dat.dtype] + ' * ' + cc.restrict_keyword + ' ' + symbol


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
    if issubclass(type(dat), cuda_base.Array):
        # Case for cuda_base.Array and cuda_data.ScalarArray.
        if not access_type.incremented:

            _s = Code(_space + get_variable_prefix(access_type) +
                                 host.ctypes_map[dat.dtype] + ' *' +
                                 symbol_internal + ' = ' + symbol_external +
                                 ';\n')

        else:

            if access_type is access.INC:
                # create tmp var
                _s = Code(_space + host.ctypes_map[dat.dtype] +
                                     ' ' + symbol_internal + '[' +
                                     str(dat.ncomp) + '] = {')

                # copy existing value into var
                for cx in range(dat.ncomp - 1):
                    _s += symbol_external + '[' + str(cx) + '], '
                _s += symbol_external + '[' + str(dat.ncomp - 1) + ']}; \n'

                # map kernel symbol to tmp var
                _s += '#undef ' + symbol_internal + '\n'
                _s += '#define ' + symbol_internal + '(x) ' + symbol_internal +\
                      '[(x)] \n'



            elif access_type is access.INC0:
                # create the temp var and init to zero.
                _s = Code(_space + host.ctypes_map[dat.dtype] +
                                     ' ' + symbol_internal + '[' +
                                     str(dat.ncomp) + '] = { 0 };\n')

                # map kernel symbol to temp var
                _s += '#undef ' + symbol_internal + '\n'
                _s += '#define ' + symbol_internal + '(x) ' + symbol_internal +\
                      '[(x)] \n'


            else:

                raise Exception("Generate mapping failure: Incremental access_type not found.")



        return _s + '\n'

    else:
        return '\n'


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


    if type(dat) is cuda_data.TypedDat:
        #case for typed dat
        _s = Code()
        # _s += _space + get_variable_prefix(access_type) + host.ctypes_map[dat.dtype] + ' *' + symbol_internal + '[' + str(_n) +'];\n'
        # _s += _space + symbol_internal + '[0] = &' + symbol_external + '[' + str(dat.ncol) + '* _TYPE_MAP[_ix]];\n'

        _s += '#undef ' + symbol_internal + '\n'

        return _s

    elif issubclass(type(dat), cuda_base.Matrix):
        # Case for cuda_base.Array and cuda_data.ScalarArray.

        _s = Code()

        if not access_type.incremented:

            return _nl

            # First pointer, the one which is associated with the thread.
            # if pair is True:
            #     _s += _space + symbol_internal + '[0] = ' + symbol_external + '+ _ix *' + str(dat.ncomp) + ' ;\n'
            # else:
            #     _s += _space + symbol_internal + ' = ' + symbol_external + '+ _ix *' + str(dat.ncomp) + ';\n'

        else:
            # Create a local copy/new row vector for the row in the particle dat.
            if access_type is access.INC:

                _s = Code(_space + host.ctypes_map[dat.dtype] +
                                     ' _tmp_' + symbol_internal + '[' +
                                     str(dat.ncomp) + '] = {')

                for cx in range(dat.ncomp - 1):
                    _s += symbol_external + '[_ix*' + str(dat.ncomp) + '+' + \
                          str(cx) + '], '

                _s += symbol_external + '[_ix*' + str(dat.ncomp) + '+' + \
                      str(dat.ncomp - 1) + ']}; \n'


            elif access_type is access.INC0:
                _s += _space + host.ctypes_map[dat.dtype] + ' _tmp_' + \
                      symbol_internal + '[' + str(dat.ncomp) + '] = { 0 };\n'

            else:
                raise Exception("Generate mapping failure: Incremental "
                                "access_type not found.")

        return _s
    else:
        return Code(' ')




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


    if type(dat) is cuda_data.TypedDat:


        #case for typed dat
        _s = Code('#undef ' + symbol_internal + _nl)
        if pair:

            _s += '#define ' + symbol_internal + '(x,y) ' + \
                 symbol_internal + '_##x(y)' + _nl
            _s += '#define ' + symbol_internal + '_0(y) ' + symbol_external + \
                  '[' + str(dat.ncol) + '* _TYPE_MAP[' + \
                  get_first_index_symbol() + ' + (y) ]]' + _nl

            _s += '#define ' + symbol_internal + '_1(y) ' + symbol_external + \
                  '[' + str(dat.ncol) + '* _TYPE_MAP[' + \
                  get_second_index_symbol() + ' + (y) ]]' + _nl

        else:

            _s += '#define ' + symbol_internal + '(x) ' + symbol_external + \
                  '[' + str(dat.ncol) + '* _TYPE_MAP[' + \
                  get_first_index_symbol() + ' + (x) ]]' + _nl

        return _s

    elif issubclass(type(dat), cuda_base.Array):

        if not access_type.incremented:

            _s = Code('#undef ' + symbol_internal + '\n')
            _s += '\n' + '#define ' + symbol_internal + '(x) ' + \
                  symbol_external + '[(x)] \n'
        else:
            _s = Code(' ')

        return _s + _nl

    elif issubclass(type(dat), cuda_base.Matrix):
        # Case for cuda_base.Matrix and cuda_data.ParticleDat.


        _ncomp = dat.ncol
        _s = Code()

        if pair:

            if not access_type.incremented:
                #_s += _space + symbol_internal + '[1] = ' + symbol_external + '+' + str(_ncomp) + '* _iy;\n'
                _s += '#undef ' + symbol_internal + _nl

                _s += '#define ' + symbol_internal + '(x,y) ' + \
                       symbol_internal + '_##x(y)' + _nl

                _s += '#define ' + symbol_internal + '_0(y) ' + symbol_external+\
                      '[' + get_first_index_symbol() + '*' + str(_ncomp) +\
                      ' + (y) ]' + _nl
                _s += '#define ' + symbol_internal + '_1(y) ' + symbol_external+\
                      '[' + get_second_index_symbol() + '*' + str(_ncomp) +\
                      ' + (y) ]' + _nl
            else:

                # create null var
                _s += _space + host.ctypes_map[dat.dtype] + ' _null_' + \
                      symbol_internal + ' = 0; \n'

                _s += '#undef ' + symbol_internal + _nl

                _s += '#define ' + symbol_internal + '(x,y) ' + \
                       symbol_internal + '_##x(y)' + _nl

                _s += '#define ' + symbol_internal + '_0(y) ' + \
                      '_tmp_' + symbol_internal + \
                      '[ (y) ]' + _nl

                _s += '#define ' + symbol_internal + '_1(y) ' + \
                      '_null_' + symbol_internal + _nl


        else:

            '''
            _s += get_variable_prefix(access_type) + \
                  host.ctypes_map[dat.dtype] + '* __restrict__ ' + \
                  symbol_internal + ' = &' + symbol_external + \
                  '[' + str(_ncomp) + '*' + get_first_index_symbol() + \
                  ']' + '; \n'
            '''
            _s += '#undef ' + symbol_internal + '\n'
            _s += '#define ' + symbol_internal + '(x) ' + symbol_external+\
                  '[' + get_first_index_symbol() + '*' + str(_ncomp) +\
                  ' + (x) ] \n'

        return _s

    else:
        raise Exception("Generate mapping failure: unknown dat type.")



def create_post_loop_map_matrices(pair=True, symbol_external=None, symbol_internal=None, dat=None, access_type=None):
    """
    ParticleDats only.

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


    if (type(dat) is cuda_data.ParticleDat) and access_type.write:
        # Case for cuda_base.Array and cuda_data.ScalarArray.
        if access_type.incremented:
            _s = Code()
            _s += _space + 'for(int _iz = 0; _iz < ' + str(dat.ncomp) + '; _iz++ ){ \n'
            _s += 2 * _space + symbol_external + '[_ix*' + str(dat.ncomp) + '+_iz] = ' + ' _tmp_' + symbol_internal + '[_iz]; \n'
            _s += _space + '}\n'

            # slooooooow
            #_s += _space + 'memcpy( &' + symbol_external + '[_ix * ' + str(dat.ncomp) + '], _tmp_' + symbol_internal + ', sizeof(' + host.ctypes_map[dat.dtype] + ')*' + str(dat.ncomp) + '); \n'

            return _s
        else:
            return Code()


    else:
        return Code()

def generate_reduction_final_stage(symbol_external, symbol_internal, dat, access_type):
    """
    Reduce arrays here

    :arg string symbol_external: variable name for shared library
    :arg string symbol_internal: variable name for kernel.
    :arg cuda_data.data: :class:`~cuda_data.ParticleDat` or :class:`~cuda_data.ScalarArray` cuda_data.object in question.
    :arg access access_type: Access being used.

    :return: string for initialisation code.
    """

    _space = ' ' * 14
    if issubclass(type(dat), cuda_base.Array) and access_type.incremented:
        # Case for cuda_base.Array and cuda_data.ScalarArray.
        # reduce on the warp.
        _s = Code() + _nl

        _s += _space + 'for(int _iz = 0; _iz < ' + str(dat.ncomp) + '; _iz++ ){ \n'
        _s += 2 * _space + symbol_internal + '[_iz] = warpReduceSumDouble(' + symbol_internal + '[_iz]); \n'
        _s += _space + '}\n'



        _s += _space + '__shared__ ' + host.ctypes_map[dat.dtype] + ' _d_red_' + symbol_internal + '[' + str(dat.ncomp) + ']; \n'
        _s += _space + 'if (  (int)(threadIdx.x & (warpSize - 1)) == 0){ \n'
        _s += 2 * _space + 'for(int _iz = 0; _iz < ' + str(dat.ncomp) + '; _iz++ ){ \n'
        _s += 3 * _space +'_d_red_' + symbol_internal + '[_iz] = 0; \n'
        _s += _space + '}} __syncthreads(); \n'



        # reduce into the shared dat.
        _s += _space + 'if (  (int)(threadIdx.x & (warpSize - 1)) == 0){ \n'
        _s += 2 * _space + 'for(int _iz = 0; _iz < ' + str(dat.ncomp) + '; _iz++ ){ \n'
        _s += 3 * _space +'atomicAddDouble(&_d_red_' + symbol_internal + '[_iz], ' + symbol_internal + '[_iz]); \n'
        _s += _space + '}} __syncthreads(); \n'

        _s += _space + 'if (threadIdx.x == 0){ \n'
        _s += 2 * _space + 'for(int _iz = 0; _iz < ' + str(dat.ncomp) + '; _iz++ ){ \n'
        _s += 3 * _space +'atomicAddDouble(&' + symbol_external + '[_iz], _d_red_' + symbol_internal + '[_iz]); \n'
        _s += _space + '}} \n'


        return _s + '\n'
    else:
        return ''




































