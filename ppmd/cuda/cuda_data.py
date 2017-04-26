"""
CUDA version of the package level data.py
"""
# System level imports
import ctypes
import numpy as np
import math
import pycuda.gpuarray as gpuarray

#package level imports
import ppmd.access as access
import ppmd.mpi as mpi
import ppmd.host as host
import ppmd.data as data
import ppmd.opt as opt

# cuda imports
import cuda_base
import cuda_build
import cuda_mpi
import cuda_runtime


_MPI = mpi.MPI
SUM = _MPI.SUM
_MPIWORLD = mpi.MPI.COMM_WORLD
_MPIRANK = mpi.MPI.COMM_WORLD.Get_rank()
_MPISIZE = mpi.MPI.COMM_WORLD.Get_size()
_MPIBARRIER = mpi.MPI.COMM_WORLD.Barrier


class ScalarArray(cuda_base.Array):

    def _init_struct(self):
         self._struct = type('ScalarArrayT',
                             (ctypes.Structure,),
                             dict(_fields_=(('ptr',
                                             ctypes.c_void_p),
                                            ('ncomp',
                                             ctypes.POINTER(ctypes.c_int)))))()

    def resize(self, new_length):
        """
        Increase the size of the array.
        :param int new_length: New array length.
        """
        if new_length > self.ncomp:
            self.realloc(new_length)

    def __call__(self, mode=access.RW, halo=False):
        return self, mode, halo

    def ctypes_data_access(self, mode=access.RW, pair=False, exchange=False):

        #print "pre", mode, self[0]
        if mode is access.INC0:
            #print self[0], mode
            self.zero()
        return self.ctypes_data

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

    def __init__(self, npart=1, ncomp=1, initial_value=None, name=None, dtype=ctypes.c_double):

        self.name = name

        self.group = None

        self.idtype = dtype
        self._ncol = ctypes.c_int(0)
        self._nrow = ctypes.c_int(0)
        self._npart_local = ctypes.c_int(0)
        self._npart = ctypes.c_int(0)

        self._ptr = ctypes.POINTER(self.idtype)()
        self._resize_callback = None

        self._dat = cuda_base._make_gpu_array(initial_value=initial_value,
                                    dtype=dtype,
                                    nrow=npart,
                                    ncol=ncomp)

        self._nrow.value = self._dat.shape[0]
        self._ncol.value = self._dat.shape[1]
        self._npart_local.value = 0

        self._vid_int = 0
        self._vid_halo = -1

        self._halo_start = ctypes.c_int(self._npart.value)
        self._npart_local_halo = ctypes.c_int(0)
        self._npart_local.value = npart
        self._npart.value = 0

        self._struct = None
        self._init_struct()

        self._1p_halo_lib = None
        self._exchange_lib = None

        self._h_mirror = cuda_base._MatrixMirror(self)

        self._norm_linf_lib = None

        self._halo_timer = opt.Timer()


    def zero(self, n=None):
        if n is None:
            self._dat.fill(0)
        else:
            cuda_runtime.LIB_CUDA_MISC['cudaMemSetZero'](
                self.ctypes_data,
                ctypes.c_int(0),
                ctypes.c_size_t(n*self.ncomp*ctypes.sizeof(self.dtype))
            )
            #self[:n:,:] = 0
        self._vid_int += 1

    def max(self):
        t = gpuarray.max(self._dat[0:self.npart_local:,:])
        return t.get()

    def norm_linf(self):
        """
        return the L1 norm of the array
        """
        val = ctypes.c_double(0)
        if self._norm_linf_lib is None:
            self._norm_linf_lib = _build_norm_linf_lib(self.dtype)

        self._norm_linf_lib(
            self.ctypes_data,
            ctypes.c_int(self.npart_local * self.ncomp),
            ctypes.byref(val)
        )
        return val.value


    def ctypes_data_access(self, mode=access.RW, pair=True, exchange=True):
        """
        :arg access mode: Access type required by the calling method.
        :return: The pointer to the data.
        """

        #print pair, self._vid_int, self._vid_halo

        if mode is access.INC0:
            self.zero(self.npart_local)
            #self.zero()

        if mode.read:
            if (self._vid_int > self._vid_halo) and pair:

                self._halo_exchange_prepare()
                if exchange:
                    self.halo_exchange(_prepare=False)


        return self.ctypes_data

    def ctypes_data_post(self, mode=access.RW):
        """
        Call after excuting a method on the data.
        :arg access mode: Access type required by the calling method.
        """
        if mode.write:
            self._vid_int += 1

    def _init_struct(self):
        self._struct = type('ParticleDatT', (ctypes.Structure,),
                            dict(_fields_=(('ptr',
                                            ctypes.c_void_p),
                                            ('nrow', ctypes.POINTER(ctypes.c_int)),
                                            ('ncol', ctypes.POINTER(ctypes.c_int)),
                                            ('npart', ctypes.POINTER(ctypes.c_int)),
                                            ('ncomp', ctypes.POINTER(ctypes.c_int)))
                                )
                            )()



    def broadcast_data_from(self, rank=0, _resize_callback=True):
        #seen from MPI_COMM_WORLD
        if _MPISIZE == 1:
            return
        else:
            s = np.array([self._nrow.value], dtype=ctypes.c_int)
            _MPIWORLD.Bcast(s, root=rank)
            self.resize(s[0], _callback=_resize_callback)
            self.npart_local = s[0]

            cuda_mpi.MPI_Bcast(_MPIWORLD,
                               self.ctypes_data,
                               ctypes.c_int(self.size),
                               ctypes.c_int(rank))


    def gather_data_on(self, rank=0):
        #seen from MPI_COMM_WORLD
        assert (rank>-1) and (rank<_MPISIZE), "Invalid mpi rank"
        if _MPISIZE == 1:
            return
        else:

            esize = ctypes.sizeof(self.idtype)

            counts = _MPIWORLD.gather(self.npart_local, root=rank)

            _ptr_new = 0

            if _MPIRANK == rank:

                _new_nloc = sum(counts)

                _new = cuda_base._create_zeros(nrow=_new_nloc,
                                               ncol=self.ncomp,
                                               dtype=self.idtype)

                _ptr_new = _new.ptr

                disp = [0] + counts[:-1:]
                disp = tuple(np.cumsum(self.ncomp * np.array(disp)))
                counts = tuple([self.ncomp*c for c in counts])

                ln = _MPISIZE
                disp_ = data.ScalarArray(dtype=ctypes.c_int, ncomp=ln)
                counts_ = data.ScalarArray(dtype=ctypes.c_int, ncomp=ln)

                disp_[:] = esize * np.array(disp)
                counts_[:] = esize * np.array(counts)

                disp = disp_
                counts = counts_

            else:
                disp = data.ScalarArray(dtype=ctypes.c_int, ncomp=_MPISIZE)
                counts = data.ScalarArray(dtype=ctypes.c_int, ncomp=_MPISIZE)

            send_count = ctypes.c_int(esize*self.npart_local*self.ncomp)

            cuda_mpi.MPI_Gatherv(_MPIWORLD,
                                 ctypes.cast(self.ctypes_data, ctypes.c_void_p),
                                 send_count,
                                 ctypes.cast(_ptr_new, ctypes.c_void_p),
                                 counts.ctypes_data,
                                 disp.ctypes_data,
                                 ctypes.c_int(rank)
                                 )

            if _MPIRANK == rank:
                self.npart_local = _new_nloc
                self._ncol.value = self.ncomp
                self._nrow.value = _new_nloc
                self._dat = _new

    def halo_start_reset(self):
        """
        Reset the starting position of the halo region in the particle dat to
         the end of the local particles.
        """
        self._halo_start.value = self.npart_local
        self._npart_local_halo.value = 0


    @property
    def struct(self):
        self._struct.ptr = self.ctypes_data
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
        return self._npart.value + self._npart_local_halo.value


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

    @property
    def npart_local_halo(self):
        return self._npart_local_halo.value

    def _halo_exchange_prepare(self):
        self._halo_timer.start()

        nproc = self.group.domain.comm.Get_size()

        if nproc == 1:
            boundary_cell_groups = self.group._halo_manager.get_boundary_cell_groups[0]
            n = self.group._halo_manager.occ_matrix.layers_per_cell * boundary_cell_groups.ncomp
            if self.max_npart < self.npart_local + n:
                self.resize(self.npart_local + n)
        else:
            self.group._halo_update_exchange_sizes()
            _total_size = self.npart_local + self.group._halo_sizes[0]
            self.resize(_total_size)


        self._halo_timer.pause()

    def halo_exchange_async(self, mode=access.RW, pair=True):
        """
        Assumes ctypes_data_access has been called
        """

        self._halo_timer.start()
        if mode.read:
            if (self._vid_int > self._vid_halo) and pair:
                self.halo_exchange(_prepare=False)

        self._halo_timer.pause()
    def halo_exchange(self, _prepare=True):

        nproc = self.group.domain.comm.Get_size()

        self._halo_timer.start()
        if _prepare:
            self._halo_exchange_prepare()

        if nproc == 1:
            self._1p_halo_exchange()
        else:
            self.halo_start_reset()

            if self._exchange_lib is None:
                self._exchange_lib = _build_exchange_lib(self)

            self._np_halo_exchange()

        self._vid_halo = self._vid_int

        self._halo_timer.pause()
        opt.PROFILE[
            'CUDA:'+self.__class__.__name__+':'+ self.name +':halo_exchange'
        ] = (self._halo_timer.time())


    def _np_halo_exchange(self):

        comm = self.group.domain.comm

        if type(self) is PositionDat:
            posdat = 1
        else:
            posdat = 0


        if (sum(self.group._halo_manager.dir_counts[:]) + self.npart_local) > self.nrow:
            self.resize(sum(self.group._halo_manager.dir_counts[:]) + self.npart_local+10)
        self._npart_local_halo.value = sum(self.group._halo_manager.dir_counts[:])

        self._exchange_lib(
            ctypes.c_int32(comm.py2f()),
            ctypes.c_int32(self.npart_local),
            ctypes.c_int32(posdat),
            ctypes.c_int32(self.group._halo_cell_max_b),
            ctypes.c_int32(self.group.get_cell_to_particle_map().layers_per_cell),
            self.group._halo_manager.get_boundary_cell_groups()[1].ctypes_data,
            self.group._halo_send_counts.ctypes_data,
            self.group._halo_manager.dir_counts.ctypes_data,
            self.group._halo_manager.get_send_ranks().ctypes_data,
            self.group._halo_manager.get_recv_ranks().ctypes_data,
            self.group._halo_b_cell_indices.ctypes_data,
            self.group.get_cell_to_particle_map().matrix.ctypes_data,
            self.group.get_cell_to_particle_map().cell_contents_count.ctypes_data,
            self.group._halo_b_scan.ctypes_data,
            self.group._halo_position_shifts.ctypes_data,
            self.ctypes_data,
            self.group._halo_tmp_space.ctypes_data
        )


    def _1p_halo_exchange(self):
        if self._1p_halo_lib is None:
            self._build_1p_halo_lib()


        self._halo_start.value = self.npart_local

        boundary_cell_groups = self.group._halo_manager.get_boundary_cell_groups[0]
        n = self.group._halo_manager.occ_matrix.layers_per_cell * boundary_cell_groups.ncomp
        n_tot = self.ncomp * n
        threads_per_block = 512
        blocksize = (ctypes.c_int * 3)(int(math.ceil(n_tot / float(threads_per_block))), 1, 1)
        threadsize = (ctypes.c_int * 3)(threads_per_block, 1, 1)


        #if self.npart_total
        #self.resize(self.npart_total)

        #print self.max_npart, self.npart, n, self.group._halo_manager.occ_matrix.layers_per_cell

        args=[blocksize,
              threadsize,
              ctypes.byref(self._halo_start),
              ctypes.c_int(n),
              ctypes.c_int(self.group._halo_manager.occ_matrix.layers_per_cell),
              self.group._halo_manager.get_boundary_cell_groups[0].struct,
              self.group._halo_manager.get_halo_cell_groups[0].struct,
              self.group._halo_manager.get_boundary_cell_to_halo_map.struct,
              self.group._halo_manager.occ_matrix.cell_contents_count.struct,
              self.group._halo_manager.occ_matrix.matrix.struct,
              self.struct]



        if type(self) == PositionDat:
            args.append(self.group._halo_manager.get_position_shifts.struct)

        self._1p_halo_lib(*args)

        self._halo_start.value += n
        self._npart_local_halo.value = n


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


        if type(self) == PositionDat:
            _hargs += ''', const cuda_Array<double> d_shifts'''
            _dargs += ''', const cuda_Array<double> d_shifts'''

            _d_call_args += ''', d_shifts'''

            # self._position_shifts = self.group._halo_manager.get_position_shifts()
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

                    d_dat.ptr[hpx* %(NCOMP)s + _comp] = d_dat.ptr[px * %(NCOMP)s + _comp] %(SHIFT_CODE)s ;

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

#########################################################################
# Build library to halo exchange a particle dat
#########################################################################


def _build_exchange_lib(dat):
    with open(str(cuda_runtime.LIB_DIR) + '/cudaHaloExchangeSource.cu','r') as fh:
        code = fh.read()
    with open(str(cuda_runtime.LIB_DIR) + '/cudaHaloExchangeSource.h','r') as fh:
        hcode = fh.read()
    assert code is not None, "Failure to read CUDA MPI packing code source"

    d = dict()
    d['DTYPE'] = host.ctypes_map[dat.dtype]
    d['NCOMP'] = dat.ncomp
    d['MPI_DTYPE'] = host.mpi_type_map[dat.dtype]

    code = code % d
    hcode = hcode % d

    return cuda_build.simple_lib_creator(hcode, code, 'ParticleDat_exchange')['cudaHaloExchangePD']


#########################################################################
# L1 Norm Lib
#########################################################################

def _build_norm_linf_lib(dtype):
    """
    Build the L1 norm lib for a ParticleDat
    """


    with open(str(cuda_runtime.LIB_DIR) + '/cudaLInfNormSource.cu','r') as fh:
        code = fh.read()
    with open(str(cuda_runtime.LIB_DIR) + '/cudaLInfNormSource.h','r') as fh:
        hcode = fh.read()
    assert code is not None, "Failure to read CUDA L inf NORM packing code source"

    d = dict()

    d['TYPENAME'] = host.ctypes_map[dtype]

    code = code % d
    hcode = hcode % d

    return cuda_build.simple_lib_creator(hcode, code, 'ParticleDat_Linf_Norm')['cudaLInfNorm']



























