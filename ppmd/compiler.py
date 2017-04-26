__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


import shlex


class Compiler(object):
    """
    Container to define different compilers.

    :arg str name: Compiler name, referance only.
    :arg str binary: Name(+path) of Compiler binary.
    :arg list c_flags: List of compile flags as strings.
    :arg list l_flags: List of link flags as strings.
    :arg list opt_flags: List of optimisation flags.
    :arg list dbg_flags: List of runtime.DEBUG flags as strings.
    :arg list compile_flag: List of compile flag as single string (eg ['-c']
    for gcc).
    :arg list shared_lib_flag: List of flags as strings to link as shared
    library.
    :arg string restrict_keyword: keyword to use for non aliased pointers
    """

    def __init__(self, name, binary, c_flags, l_flags, opt_flags, dbg_flags,
                 compile_flag, shared_lib_flag, restrict_keyword=''):


        if type(c_flags) is str:
            c_flags = shlex.split(c_flags)
        if type(l_flags) is str:
            l_flags = shlex.split(l_flags)
        if type(opt_flags) is str:
            opt_flags = shlex.split(opt_flags)
        if type(dbg_flags) is str:
            dbg_flags = shlex.split(dbg_flags)
        if type(compile_flag) is str:
            compile_flag = shlex.split(compile_flag)
        if type(shared_lib_flag) is str:
            shared_lib_flag = shlex.split(shared_lib_flag)

        self._name = name
        self._binary = binary
        self._cflags = c_flags
        self._lflags = l_flags
        self._optflags = opt_flags
        self._dbgflags = dbg_flags
        self._compileflag = compile_flag
        self._sharedlibf = shared_lib_flag
        self._restrictkeyword = restrict_keyword




    def __str__(self):
        nl = ', '
        return \
            str( self._name ) + nl + \
            str( self._binary ) + nl + \
            str( self._cflags ) + nl + \
            str( self._lflags ) + nl + \
            str( self._optflags ) + nl + \
            str( self._dbgflags ) + nl + \
            str( self._compileflag ) + nl + \
            str( self._sharedlibf ) + nl + \
            str( self._restrictkeyword )

    def __repr__(self):
        return str(self)

    @property
    def restrict_keyword(self):
        return self._restrictkeyword

    @property
    def name(self):
        """Return Compiler name."""
        return self._name

    @property
    def binary(self):
        """Return Compiler binary."""
        return self._binary

    @property
    def c_flags(self):
        """Return Compiler compile flags"""
        return self._cflags

    @property
    def l_flags(self):
        """Return Compiler link flags"""
        return self._lflags

    @property
    def opt_flags(self):
        """Return Compiler runtime.DEBUG flags"""
        return self._optflags

    @property
    def dbg_flags(self):
        """Return Compiler runtime.DEBUG flags"""
        return self._dbgflags

    @property
    def compile_flag(self):
        """Return Compiler compile flag."""
        return self._compileflag

    @property
    def shared_lib_flag(self):
        """Return Compiler link as shared library flag."""
        return self._sharedlibf