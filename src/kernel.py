import hashlib
import os
import constant

class Kernel(object):
    '''Computational kernel, i.e. C-code + numerical constants.

    Stores the C source code of a kernel and substitutes in any 
    numerical constants that are to be copied it when the class is
    created.

    :arg name: name of the kernel
    :arg code: C source code
    :arg constants: List of constants (type :class:`.Constant`) which 
        are to be substituted in
    '''
    def __init__(self,name,code,constants):
        self._name = name
        self._code = code
        for x in constants:
            self._code = x.replace(self._code)

    def hexdigest(self):
        '''Unique md5 hexdigest which is used for identifying the kernel.'''
        m = hashlib.md5()
        m.update(self._code)
        return m.hexdigest()

    @property
    def name(self):
        '''Kernel name.'''
        return self._name
    
    @property
    def code(self):
        '''Kernel source code after substitution of numerical constants'''
        return self._code
    
