import hashlib
import os
import constant
import re

class Kernel(object):
    '''Computational kernel, i.e. C-code + numerical constants.

    Stores the C source code of a kernel and substitutes in any 
    numerical constants that are to be copied it when the class is
    created.

    :arg name: name of the kernel
    :arg code: C source code
    :arg constants: List of constants (type :class:`.Constant`) which are to be substituted in.
    :arg reductions: list of reductions required by kernel if using parallel looping. 
    '''
    def __init__(self,name,code,constants=None,headers=None,reductions=None):
        self._name = name
        self._code = code
        self._headers = headers
        self._reductions = reductions
        self._reduction_dict = {}
        
        if (constants!=None):
            for x in constants:
                self._code = x.replace(self._code)
        if (self._reductions!=None):
            for x in self._reductions:
                    self._reduction_dict[x.variable]=x
        

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
    
    @property
    def headers(self):
        '''Return C headers required for kernel'''
        return self._headers 
        
    def reduction_variable_lookup(self, var):
        '''Provides a method to determine if a variable undergoes a reduction.'''
        return self._reduction_dict.get(var)
        

        

    
class reduction(object):
    '''
    Object to store reduction operations required by kernels. Currently holds only one reduction per instance.
    
    :arg str variable: Variable name eg 'U'.
    :arg str pointer: C pointer syntax of variable being reduced upon eg 'U[0]'.
    :arg char operator: operator performed in reduction, default '+'.
    '''
    def __init__(self,variable,pointer,operator = '+'):
        self._var = variable
        self._pointer = pointer
        self._op = operator
        
    @property
    def variable(self):
        '''Returns variable name '''
        return self._var
    @property
    def pointer(self):
        '''Returns C pointer syntax'''
        return self._pointer
    @property
    def operator(self):
        '''Returns C operator'''
        return self._op
    @property
    def index(self):
        '''Returns index in C pointer syntax, eg U[0] returns 0'''
        return re.match('('+self._var+'\[)(.*)(\])',self._pointer).group(2)
    
    
    
    
    
    
