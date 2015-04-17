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
    def __init__(self,name,code,constants=None,headers=None, omp_methods = None):
        self._name = name
        self._code = code
        self._headers = headers
        self._omp_methods = omp_methods
        
        
        if (constants!=None):
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
    
    @property
    def headers(self):
        '''Return C headers required for kernel'''
        return self._headers

    @property
    def OpenMPInitStr(self):
        if (self._omp_methods!=None):
            return self._omp_methods.init
        
    @property
    def OpenMPDecStr(self):
        if (self._omp_methods!=None):
            return self._omp_methods.dec
        
    @property
    def OpenMPFinalStr(self):
        if (self._omp_methods!=None):                     
            return self._omp_methods.final 
            
class OpenMPMethod(object):
    def __init__(self, init_str, declaration_str, finalise_str): 
        self._init_str = init_str
        self._dec_str = declaration_str
        self._final_str = finalise_str
    
    @property    
    def init(self):
        return self._init_str
    
    @property    
    def dec(self):
        return self._dec_str    
    
    @property    
    def final(self):
        return self._final_str    
    
    
    
    
    
    
    
    
    
    
    
    
