class compiler(object):
    def __init__(self,name,binary,cflags,lflags,dbgflags,compileflag,sharedlibraryflag):
        self._name = name
        self._binary = binary
        self._cflags = cflags
        self._lflags = lflags
        self._dbgflags = dbgflags
        self._compileflag = compileflag
        self._sharedlibf = sharedlibraryflag
    @property    
    def name(self):
        return self._name
    @property
    def binary(self):
        return self._binary
    @property
    def cflags(self):
        return self._cflags
    @property
    def lflags(self):
        return self._lflags
    @property
    def dbgflags(self):
        return self._dbgflags
    @property
    def compileflag(self):
        return self._compileflag
    @property
    def sharedlibflag(self):
        return self._sharedlibf            
        
GCC = compiler(['GCC'],['gcc'],['-O3','-fpic','-std=c99'],['-lm'],['-g'],['-c'],['-shared'])

GCC_OpenMP = compiler(['GCC'],['gcc'],['-O3','-fpic','-fopenmp','-lgomp','-lpthread','-lc','-lrt','-std=c99'],['-fopenmp','-lgomp','-lpthread','-lc','-lrt'],['-g'],['-c'],['-shared'])

ICC = compiler(['ICC'],['icc'],['-O3','-fpic','-std=c99'],['-lm'],['-g'],['-c'],['-shared'])

ICC_OpenMP = compiler(['ICC'],['icc'],['-O3','-fpic','-openmp','-lgomp','-lpthread','-lc','-lrt','-std=c99'],['-openmp','-lgomp','-lpthread','-lc','-lrt'],['-g'],['-c'],['-shared'])


