import hashlib
import re
import cgen

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
        """Replace all occurances in a string and return result

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














