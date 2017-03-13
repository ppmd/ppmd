__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import hashlib
import re
import cgen
import ctypes
import host
import numpy as np

def analyse(kernel_in=None, dat_dict=None):
    """
    :arg kernel kernel_in: Kernel to analyse.
    :arg list dat_dict: List of variables which should be assumed are in global memory.
    :returns: Dict of operations and their estimated number of occurences.
    """
    assert kernel_in is not None, "kernel.analyse error: No kernel passed"
    assert dat_dict is not None, "kernel.analsyse error: No symbols passed to use for global memory"
    
    _code = kernel_in.code

    print _code
    
    _ops_lookup = ['+', '-', '*', '/']
    _ops_obs = dict((op, 0) for op in _ops_lookup)
    
    for c in _code:
        if c in _ops_lookup:
            _ops_obs[c] += 1

    return _ops_obs

class Kernel(object):
    """Computational kernel, i.e. C-code + numerical constants.

    Stores the C source code of a kernel and substitutes in any 
    numerical constants that are to be copied it when the class is
    created.

    :arg name: name of the kernel
    :arg code: C source code
    :arg constants: List of constants (type :class:`.Constant`) which are to be substituted in.
    :arg reductions: list of reductions required by kernel if using parallel looping. 
    """

    def __init__(self, name, code, constants=None, headers=None, reductions=None, static_args=None):
        self._name = name
        self._code = code
        self._headers = headers
        self._reductions = reductions
        self._sargs = static_args

        self._reduction_dict = {}

        if constants is not None:
            for x in constants:
                self._code = x.replace(self._code)
        if self._reductions is not None:
            for x in self._reductions:
                self._reduction_dict[x.variable] = x

    def hexdigest(self):
        """Unique md5 hexdigest which is used for identifying the kernel."""
        m = hashlib.md5()
        m.update(self._code)
        return m.hexdigest()

    @property
    def name(self):
        """Kernel name."""
        return self._name

    @property
    def code(self):
        """Kernel source code after substitution of numerical constants"""
        return self._code

    @property
    def headers(self):
        """Return C headers required for kernel"""
        return self._headers

    def reduction_variable_lookup(self, var):
        """Provides a method to determine if a variable undergoes a Reduction."""
        return self._reduction_dict.get(var)

    @property
    def static_args(self):
        return self._sargs


class Reduction(object):
    """
    Object to store Reduction operations required by kernels. Currently holds only one Reduction per instance.
    
    :arg str variable: Variable name eg 'u'.
    :arg str pointer: C pointer syntax of variable being reduced upon eg 'u[0]'.
    :arg char operator: operator performed in Reduction, default '+'.
    """

    def __init__(self, variable, pointer, operator='+'):
        self._var = variable
        self._pointer = pointer
        self._op = operator

    @property
    def variable(self):
        """Returns variable name """
        return self._var

    @property
    def pointer(self):
        """Returns C pointer syntax"""
        return self._pointer

    @property
    def operator(self):
        """Returns C operator"""
        return self._op

    @property
    def index(self):
        """Returns index in C pointer syntax, eg u[0] returns 0"""
        return re.match('(' + self._var + '\[)(.*)(\])', self._pointer).group(2)

class Constant(object):
    """Class representing a numerical constant.

    This class can be used to use placeholders for constants
    in kernels.

    :arg str name: Name of constant
    :arg value: Numerical value (can actually be any data type)
    """

    def __init__(self, name, value):
        self._name = name
        self._value = value

    def replace(self, s):
        """Replace all occurrences in a string and return result

        Ignores the constant if it is not a C-variable. For example,
        if the name of the constant is ``mass``, then it would not replace
        it in ``mass1`` or ``Hmass``.

        :arg str s: string to work on
        """

        # forbiddenChars='[^a-zA-Z0-9_]' #='[\W]'='[^\w]'

        forbiddenchars = '[\W]'
        regex = '(?<=' + forbiddenchars + ')(' + self._name + ')(?=' + forbiddenchars + ')'

        return re.sub(regex, str(repr(self._value)), s)




class Header(object):
    def __init__(self, name, system=True):
        """
        :param str name: Name of header file to include.
        :param bool system: True if header is a system header
        """
        self._name = name
        self.ast = cgen.Include(name, system)

    def __str__(self):
        return self._name





class Module(object):

    def get_cpp_headers_ast(self):
        """
        Return the code to include the required header file(s).
        """
        pass

    def get_cpp_arguments_ast(self):
        """
        Return the code to define arguments to add to the library.
        """
        pass

    def get_cpp_pre_loop_code_ast(self):
        """
        Return the code to place before the loop.
        """
        pass

    def get_cpp_post_loop_code_ast(self):
        """
        Return the code to place after the loop.
        """
        pass

    def get_python_parameters(self):
        """
        Return the parameters to add to the launch of the shared library.
        """
        pass




class Cpp11MT19937(Module):

    def __init__(self):
    
        pass
        #lib = build.simple_lib_creator(_ex_header,
        #    _ex_code,
        #    'HALO_EXCHANGE_PD',
        #    CC=build.MPI_CC
        #)['HALO_EXCHANGE_PD']

    def get_cpp_headers_ast(self):
        """
        Return the code to include the required header file(s).
        """
        return cgen.Include('chrono')


    def get_cpp_arguments_ast(self):
        """
        Return the code to define arguments to add to the library.
        """
        return cgen.Pointer(cgen.Value(host.double_str, '_loop_timer_return'))

    def get_cpp_pre_loop_code_ast(self):
        """
        Return the code to place before the loop.
        """
        _s = 'std::chrono::high_resolution_clock::time_point _loop_timer_t0 ='\
             ' std::chrono::high_resolution_clock::now(); \n'
        return cgen.Module([cgen.Line(_s)])


    def get_cpp_post_loop_code_ast(self):
        """
        Return the code to place after the loop.
        """
        _s = 'std::chrono::high_resolution_clock::time_point _loop_timer_t1 ='\
             ' std::chrono::high_resolution_clock::now(); \n' \
             ' std::chrono::duration<double> _loop_timer_res = _loop_timer_t1'\
             ' - _loop_timer_t0; \n' \
             '*_loop_timer_return += (double) _loop_timer_res.count(); \n'
        return cgen.Module([cgen.Line(_s)])


    def get_python_parameters(self):
        """
        Return the parameters to add to the launch of the shared library.
        """
        return ctypes.byref(self._time)









