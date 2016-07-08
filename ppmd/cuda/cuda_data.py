"""
CUDA version of the package level data.py
"""
# System level imports
import ctypes
import numpy as np
import math

#package level imports
import ppmd.access as access
import ppmd.mpi as mpi
import ppmd.host as host
import ppmd.data as data

# cuda imports
import cuda_base
import cuda_halo
import cuda_build
import cuda_mpi

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


        self._h_mirror = cuda_base._ArrayMirror(self)

    def __getitem__(self, key):
        self._h_mirror.copy_from_device()
        return self._h_mirror.mirror.data[key]

    def __setitem__(self, key, value):
        self._h_mirror.copy_from_device()
        self._h_mirror.mirror.data[key] = value
        self._h_mirror.copy_to_device()
        self._version += 1

    def __repr__(self):
        return str(self.__getitem__(slice(None, None, None)))
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
        self._resize_callback = None

        if initial_value is not None:
            if type(initial_value) is np.ndarray:
                self._create_from_existing(initial_value, dtype)
                self._npart = ctypes.c_int(self._nrow.value)

            elif type(initial_value) is data.ParticleDat:
                self._create_from_existing(initial_value.data, initial_value.dtype)
                self._npart = ctypes.c_int(initial_value.npart_local)

            else:
                self._create_from_existing(np.array([initial_value]),dtype)
                self._npart = ctypes.c_int(self._nrow.value)

        else:
            if max_npart is not None:
                self._nrow = ctypes.c_int(max_npart)
            else:
                self._nrow = ctypes.c_int(npart)

            self._create_zeros(self._nrow.value, ncomp, dtype)
            self._npart = ctypes.c_int(npart)

        self._vid_int = 0

        self._halo_start = ctypes.c_int(self._npart.value)
        self._npart_halo = ctypes.c_int(0)
        self.npart_local = npart

        self._struct = type('ParticleDatT', (ctypes.Structure,),
                            dict(_fields_=(('ptr',
                                            ctypes.POINTER(self.idtype)),
                                            ('nrow', ctypes.POINTER(ctypes.c_int)),
                                            ('ncol', ctypes.POINTER(ctypes.c_int)),
                                            ('npart', ctypes.POINTER(ctypes.c_int)),
                                            ('ncomp', ctypes.POINTER(ctypes.c_int)))
                                )
                            )()

        self._1p_halo_lib = None

        self._h_mirror = cuda_base._MatrixMirror(self)




    def broadcast_data_from(self, rank=0, _resize_callback=True):

        if mpi.MPI_HANDLE.nproc == 1:
            return
        else:
            s = np.array([self._nrow.value], dtype=ctypes.c_int)
            mpi.MPI_HANDLE.comm.Bcast(s, root=rank)
            self.resize(s[0], _callback=_resize_callback)


            cuda_mpi.MPI_Bcast(self.ctypes_data,
                               ctypes.c_int(self.size),
                               ctypes.c_int(rank))





    def halo_start_reset(self):
        """
        Reset the starting position of the halo region in the particle dat to
         the end of the local particles.
        """
        self._halo_start.value = self.npart_local
        self._npart_halo.value = 0


    def __getitem__(self, key):
        self._h_mirror.copy_from_device()
        return self._h_mirror.mirror.data[key]

    def __setitem__(self, key, value):
        self._h_mirror.copy_from_device()
        self._h_mirror.mirror.data[key] = value
        self._h_mirror.copy_to_device()
        self._vid_int += 1

    def __repr__(self):
        return str(self.__getitem__(slice(None, None, None)))


    @property
    def struct(self):
        self._struct.ptr = self._ptr
        self._struct.nrow = ctypes.pointer(self._nrow)
        self._struct.ncol = ctypes.pointer(self._ncol)
        self._struct.npart_local = ctypes.pointer(self._npart)
        self._struct.ncomp = ctypes.pointer(self._ncol)
        return self._struct

    @property
    def max_npart(self):
        return self._nrow.value


    @property
    def npart_total(self):
        return self._npart.value + self._npart_halo.value


    def resize(self, n, _callback=True):
        if _callback and (self._resize_callback is not None):
            self._resize_callback(n)
            return

        if n > self.max_npart:
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

        boundary_cell_groups = cuda_halo.HALOS.get_boundary_cell_groups[0]
        n = cuda_halo.HALOS.occ_matrix.layers_per_cell * boundary_cell_groups.ncomp
        if self.max_npart < self.npart_local + n:
            self.resize(self.npart_local + n)

        self._halo_start.value = self.npart_local

        n_tot = self.ncomp * n
        threads_per_block = 512
        blocksize = (ctypes.c_int * 3)(int(math.ceil(n_tot / float(threads_per_block))), 1, 1)
        threadsize = (ctypes.c_int * 3)(threads_per_block, 1, 1)


        #if self.npart_total
        #self.resize(self.npart_total)

        #print self.max_npart, self.npart, n, cuda_halo.HALOS.occ_matrix.layers_per_cell

        args=[blocksize,
              threadsize,
              ctypes.byref(self._halo_start),
              ctypes.c_int(n),
              ctypes.c_int(cuda_halo.HALOS.occ_matrix.layers_per_cell),
              cuda_halo.HALOS.get_boundary_cell_groups[0].struct,
              cuda_halo.HALOS.get_halo_cell_groups[0].struct,
              cuda_halo.HALOS.get_boundary_cell_to_halo_map.struct,
              cuda_halo.HALOS.occ_matrix.cell_contents_count.struct,
              cuda_halo.HALOS.occ_matrix.matrix.struct,
              self.struct]



        if self.name == 'positions':
            args.append(cuda_halo.HALOS.get_position_shifts.struct)

        self._1p_halo_lib(*args)

        self._halo_start.value += n
        self._npart_halo.value = n


    def _build_1p_halo_lib(self):

        _name = '_1p_halo_lib'

        _hargs = '''const int blocksize[3],
                    const int threadsize[3],
                    int * h_n_total,
                    const int h_n,
                    const int h_npc,
                    const cuda_Array<int> d_b,
                    const cuda_Array<int> d_h,
                    const cuda_Array<int> d_bhc_map,
                    cuda_Array<int> d_ccc,
                    cuda_Matrix<int> d_occ_matrix,
                    cuda_ParticleDat<%(TYPE)s> d_dat
                   ''' % {'TYPE': host.ctypes_map[self.idtype]}

        _dargs = '''const cuda_Array<int> d_b,
                    const cuda_Array<int> d_h,
                    const cuda_Array<int> d_bhc_map,
                    cuda_Array<int> d_ccc,
                    cuda_Matrix<int> d_occ_matrix,
                    cuda_ParticleDat<%(TYPE)s> d_dat
                    ''' % {'TYPE': host.ctypes_map[self.idtype]}

        _d_call_args = '''d_b, d_h, d_bhc_map, d_ccc, d_occ_matrix, d_dat'''


        if self.name == 'positions':
            _hargs += ''', const cuda_Array<double> d_shifts'''
            _dargs += ''', const cuda_Array<double> d_shifts'''

            _d_call_args += ''', d_shifts'''

            # self._position_shifts = cuda_halo.HALOS.get_position_shifts()
            _shift_code = ''' + d_shifts.ptr[d_bhc_map.ptr[_cx]*3 + _comp]'''
            _occ_code = '''
            d_occ_matrix.ptr[ d_npc * d_h.ptr[_cx] + _pi ] = hpx;

            // if particle layer is zero write the cell contents count.
            if (_pi == 0){
                d_ccc.ptr[d_h.ptr[_cx]] = d_ccc.ptr[d_b.ptr[_cx]];
            }

            '''
        else:
            _shift_code = ''''''
            _occ_code = ''''''

        _header = '''
            #include <cuda_generic.h>
            extern "C" int %(NAME)s(%(HARGS)s);
        ''' % {'NAME': _name, 'HARGS': _hargs}

        _src = '''

        __constant__ int d_n_total;
        __constant__ int d_n;
        __constant__ int d_npc;

        __global__ void d_1p_halo_copy_shift(%(DARGS)s){

            //particle index
            const int idx = (threadIdx.x + blockIdx.x*blockDim.x)/%(NCOMP)s;


            if (idx < d_n){
                //component corresponding to thread.
                const int _comp = (threadIdx.x + blockIdx.x*blockDim.x) %% %(NCOMP)s;

                const int _cx = idx/d_npc;
                const int _bc = d_b.ptr[_cx]; // some boundary cell
                const int _pi = idx %% d_npc; // particle layer


                if (_pi < d_ccc.ptr[_bc]){ //Do we need this thread to do anything?

                    // local index of particle
                    const int px = d_occ_matrix.ptr[_bc*d_npc + _pi];

                    //halo index of particle
                    const int hpx = d_n_total + idx;

                    d_dat.ptr[hpx* %(NCOMP)s + _comp] = d_dat.ptr[px * %(NCOMP)s + _comp] + %(SHIFT_CODE)s ;

                    //printf("hpx %%d, px %%d, _cx %%d, _bc %%d halo %%d \\n", hpx, px, _cx, _bc, d_bhc_map.ptr[_cx]);

                    %(OCC_CODE)s

                    //printf("shift %%f, halo %%d, _comp %%d \\n", d_shifts.ptr[d_bhc_map.ptr[_cx]*3 + _comp], d_bhc_map.ptr[_cx], _comp);

                    }


            }
            return;


        }

        int %(NAME)s(%(HARGS)s){
            checkCudaErrors(cudaMemcpyToSymbol(d_n_total, h_n_total, sizeof(int)));
            checkCudaErrors(cudaMemcpyToSymbol(d_n, &h_n, sizeof(h_n)));
            checkCudaErrors(cudaMemcpyToSymbol(d_npc, &h_npc, sizeof(h_npc)));

            dim3 bs; bs.x = blocksize[0]; bs.y = blocksize[1]; bs.z = blocksize[2];
            dim3 ts; ts.x = threadsize[0]; ts.y = threadsize[1]; ts.z = threadsize[2];

            d_1p_halo_copy_shift<<<bs,ts>>>(%(D_C_ARGS)s);
            checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError("1proc halo lib Execution failed. \\n");

            return 0;
        }
        ''' % {'NAME': _name,
               'HARGS': _hargs,
               'DARGS': _dargs,
               'D_C_ARGS': _d_call_args,
               'NCOMP': self.ncomp,
               'SHIFT_CODE': _shift_code,
               'OCC_CODE': _occ_code}




        self._1p_halo_lib = cuda_build.simple_lib_creator(_header, _src, _name)[_name]



#########################################################################
# PositionDat.
#########################################################################

class PositionDat(ParticleDat):
    pass





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

        self._vid_int = 0

        self._struct = type('MatrixT', (ctypes.Structure,), dict(_fields_=(('ptr', ctypes.POINTER(self.idtype)),
                                                                          ('nrow', ctypes.POINTER(ctypes.c_int)),
                                                                          ('ncol', ctypes.POINTER(ctypes.c_int)))))()

        self.name = name

    def __call__(self, mode=access.RW, halo=False):
        return self, mode, halo




