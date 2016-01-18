"""
CUDA version of the package level data.py
"""
# System level imports
import ctypes
import numpy as np

#package level imports
import ppmd.access as access
import ppmd.mpi as mpi

# cuda imports
import cuda_base
import cuda_halo
import cuda_build


class ScalarArray(cuda_base.Array):

    def __init__(self, initial_value=None, name=None, ncomp=0, dtype=ctypes.c_double):

        self.name = name

        self.idtype = dtype
        self._ncomp = ctypes.c_int(0)
        self._ptr = ctypes.POINTER(self.idtype)()

        if initial_value is not None:
            if type(initial_value) is np.ndarray:
                self._create_from_existing(initial_value, dtype)
            else:
                self._create_from_existing(np.array([initial_value]), dtype)
        else:
            self._create_zeros(ncomp, dtype)

        self._version = 0
        self._struct = type('ScalarArrayT', (ctypes.Structure,), dict(_fields_=(('ptr', ctypes.POINTER(self.idtype)), ('ncomp', ctypes.POINTER(ctypes.c_int)))))()

    @property
    def struct(self):
        self._struct.ptr = self._ptr
        self._struct.ncomp = ctypes.pointer(self._ncomp)
        return self._struct

    @property
    def version(self):
        """
        Get the version of this array.
        :return int version:
        """
        return self._version

    def resize(self, new_length):
        """
        Increase the size of the array.
        :param int new_length: New array length.
        """
        if new_length > self.ncomp:
            self.realloc(new_length)

    def __call__(self, mode=access.RW, halo=False):
        return self, mode, halo

###################################################################################################
# Blank arrays.
###################################################################################################

NullIntScalarArray = ScalarArray(dtype=ctypes.c_int)
"""Empty integer :class:`~data.ScalarArray` for specifying a kernel argument that may not yet be
declared."""

NullDoubleScalarArray = ScalarArray(dtype=ctypes.c_double)
"""Empty double :class:`~data.ScalarArray` for specifying a kernel argument that may not yet be
declared."""


###################################################################################################
# cuda ParticleDat
###################################################################################################


class ParticleDat(cuda_base.Matrix):
    """
    Cuda particle dat
    """
    def __init__(self, npart=0, ncomp=0, initial_value=None, name=None, dtype=ctypes.c_double, max_npart=None):

        self.name = name

        self.idtype = dtype
        self._ncol = ctypes.c_int(0)
        self._nrow = ctypes.c_int(0)

        self._ptr = ctypes.POINTER(self.idtype)()

        if initial_value is not None:
            if type(initial_value) is np.ndarray:
                self._create_from_existing(initial_value, dtype)
            else:
                self._create_from_existing(np.array([initial_value]),dtype)

            self._max_npart = ctypes.c_int(self._nrow.value)
            self._npart = ctypes.c_int(self._nrow.value)

        else:
            if max_npart is not None:
                self.max_npart = ctypes.c_int(max_npart)
            else:
                self.max_npart = ctypes.c_int(npart)

            self._create_zeros(self.max_npart.value, ncomp, dtype)
            self._npart = ctypes.c_int(npart)

        self._ncomp = ctypes.c_int(self._ncol.value)


        self._version = 0

        self._halo_start = ctypes.c_int(self._npart.value)
        self._npart_halo = ctypes.c_int(0)

        self._struct = type('ParticleDatT', (ctypes.Structure,), dict(_fields_=(('ptr', ctypes.POINTER(self.idtype)),
                                                                                ('nrow', ctypes.POINTER(ctypes.c_int)),
                                                                                ('ncol', ctypes.POINTER(ctypes.c_int)),
                                                                                ('npart', ctypes.POINTER(ctypes.c_int)),
                                                                                ('ncomp', ctypes.POINTER(ctypes.c_int)))))()

        self._1p_halo_lib = None

    @property
    def struct(self):
        self._struct.ptr = self._ptr
        self._struct.nrow = ctypes.pointer(self.max_npart)
        self._struct.ncol = ctypes.pointer(self._ncol)
        self._struct.npart = ctypes.pointer(self._npart)
        self._struct.ncomp = ctypes.pointer(self._ncomp)
        return self._struct

    @property
    def npart_total(self):
        return self._npart.value + self._npart_halo.value


    def resize(self, n):
        if n > self.max_npart:
            self.max_npart = n
            self.realloc(n, self.ncol)

    def __call__(self, mode=access.RW, halo=True):
        return self, mode, halo

    @property
    def ncomp(self):
        return self._ncol.value

    @property
    def npart(self):
        return self._npart.value

    def halo_exchange(self):
        if mpi.MPI_HANDLE.nproc == 1:
            self._1p_halo_exchange()

    def _1p_halo_exchange(self):
        if self._1p_halo_lib is None:
            self._build_1p_halo_lib()



    def _build_1p_halo_lib(self):

        _name = '1p_halo_lib'

        _hargs = '''const int blocksize[3],
                   const int threadsize[3],
                   const int h_n,               // Total possible number of cells*maximum number of layers in use.
                   const int h_npc,             // Maximum number of layers in use
                   '''

        _dargs = '''
                    '''

        _d_call_args = ''''''


        if self.name == 'positions':
            _hargs += '''const double* %(R)s d_shifts \n''' % {'R':cuda_build.NVCC.restrict_keyword}
            _dargs += '''const double* %(R)s d_shifts \n''' % {'R':cuda_build.NVCC.restrict_keyword}

            _d_call_args += ''' d_shifts'''

            self._position_shifts = cuda_halo.HALOS.get_position_shifts()
            _shift_code = ''''''
        else:
            _shift_code = ''''''

        _header = '''
            #include <cuda_generic.h>
            extern "C" int %(NAME)s(%(HARGS)s);
        ''' % {'NAME': _name, 'HARGS': _hargs}

        _src = '''

        __constant__ int d_n;
        __constant__ int d_npc;

        __global__ void d_1p_halo_copy_shift(%(DARGS)s){

            return;
        }

        int %(NAME)s(%(HARGS)s){
            checkCudaErrors(cudaMemcpyToSymbol(d_n, &h_n, sizeof(h_n)));
            checkCudaErrors(cudaMemcpyToSymbol(d_npc, &h_npc, sizeof(h_npc)));

            dim3 bs; bs.x = blocksize[0]; bs.y = blocksize[1]; bs.z = blocksize[2];
            dim3 ts; ts.x = threadsize[0]; ts.y = threadsize[1]; ts.z = threadsize[2];

            d_1p_halo_copy_shift<<<bs,ts>>>(%(DARGS)s);
            checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError("1proc halo lib Execution failed. \\n");

            return 0;
        }
        '''




        self._1p_halo_lib = cuda_build.simple_lib_creator(_header, _src, _name)









class TypedDat(cuda_base.Matrix):
    # Follows cuda_base.Matrix Init except for name prameter/attribute.

    def __init__(self, nrow=0, ncol=0, initial_value=None, name=None, dtype=ctypes.c_double):

        self.idtype = dtype


        self._ncol = ctypes.c_int(0)
        self._nrow = ctypes.c_int(0)

        self._ptr = ctypes.POINTER(self.idtype)()

        if initial_value is not None:
            if type(initial_value) is np.ndarray:
                self._create_from_existing(initial_value, dtype)
            else:
                self._create_from_existing(np.array([initial_value]),dtype)

        else:
            self._create_zeros(nrow, ncol, dtype)

        self._version = 0

        self._struct = type('MatrixT', (ctypes.Structure,), dict(_fields_=(('ptr', ctypes.POINTER(self.idtype)),
                                                                          ('nrow', ctypes.POINTER(ctypes.c_int)),
                                                                          ('ncol', ctypes.POINTER(ctypes.c_int)))))()

        self.name = name

    def __call__(self, mode=access.RW, halo=False):
        return self, mode, halo




