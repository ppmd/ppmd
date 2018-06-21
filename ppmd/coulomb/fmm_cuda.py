from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


from math import log, ceil
from ppmd.coulomb.octal import *
import numpy as np
from ppmd import runtime, host, kernel, pairloop, data, access, mpi, opt
from ppmd.lib import build
import ctypes
import os
import math
import cmath
from threading import Thread, Lock
from scipy.special import lpmv, rgamma, gammaincc, lambertw

from ppmd.cuda import *

from ppmd.access import *
from ppmd.data import ParticleDat

REAL = ctypes.c_double
INT64 = ctypes.c_int64

def red(input):
    try:
        from termcolor import colored
        return colored(input, 'red')
    except Exception as e: return input
def green(input):
    try:
        from termcolor import colored
        return colored(input, 'green')
    except Exception as e: return input
def yellow(input):
    try:
        from termcolor import colored
        return colored(input, 'yellow')
    except Exception as e: return input

def _numpy_ptr(arr):
    return arr.ctypes.data_as(ctypes.c_void_p)

def _check_dtype(arr, dtype):
    if arr.dtype != dtype:
        raise RuntimeError('Bad data type. Expected: {} Found: {}.'.format(
            str(dtype), str(arr.dtype)))
    if issubclass(type(arr), np.ndarray): return _numpy_ptr(arr)
    elif issubclass(type(arr), host.Matrix): return arr.ctypes_data
    elif issubclass(type(arr), host.Array): return arr.ctypes_data
    elif issubclass(type(arr), cuda_base.gpuarray.GPUArray):
        return ctypes.cast(arr.ptr, ctypes.c_void_p)
    else: raise RuntimeError('unknown array type passed: {}'.format(type(arr)))


_SRC_DIR = os.path.dirname(os.path.realpath(__file__))

np.set_printoptions(threshold=np.nan)

class TranslateMTLCuda(object):
    def __init__(self, dtype, tree, nlevel, a_arr, ar_arr, p_arr, e_arr,
                 int_list, int_tlookup, int_plookup, int_radius, ipower_mtl,
                 wigner_f, wigner_b, arn0):
        self.tree = tree
        self.L = nlevel
        ncomp = (self.L**2) * 2
        self.tree_plain = OctalCudaDataTree(tree=tree, mode='plain',
                                            dtype=dtype, ncomp=ncomp)
        self.tree_halo = OctalCudaDataTree(tree=tree, mode='halo',
                                           dtype=dtype, ncomp=ncomp)



        self._d_a = cuda_base.gpuarray.to_gpu(a_arr)
        self._d_ar = cuda_base.gpuarray.to_gpu(ar_arr)
        self._d_p = cuda_base.gpuarray.to_gpu(p_arr)
        self._d_e = cuda_base.gpuarray.to_gpu(e_arr)

        self._int_list = []
        for lx in int_list:
            if lx is not None:
                ne = cuda_base.gpuarray.to_gpu(lx)
            else:
                ne = None
            self._int_list.append(ne)

        self._d_int_tlookup = cuda_base.gpuarray.to_gpu(int_tlookup)
        self._d_int_plookup = cuda_base.gpuarray.to_gpu(int_plookup)
        self._d_int_radius = cuda_base.gpuarray.to_gpu(int_radius)

        self._ipower_mtl = cuda_base.gpuarray.to_gpu(ipower_mtl)

        jlookup = np.zeros(ncomp, dtype=INT64)
        klookup = np.zeros(ncomp, dtype=INT64)

        ind = 0
        for jx in range(nlevel):
            for kx in range(-1*jx, jx+1):
                jlookup[ind] = jx
                klookup[ind] = kx
                ind += 1

        self._jlookup = cuda_base.gpuarray.to_gpu(jlookup)
        self._klookup = cuda_base.gpuarray.to_gpu(klookup)




        # need tmp space to rotate moments
        self.tmp_plain0 = OctalCudaDataTree(tree=tree, mode='plain',
                                            dtype=dtype, ncomp=ncomp)
        self.tmp_plain1 = OctalCudaDataTree(tree=tree, mode='plain',
                                            dtype=dtype, ncomp=ncomp)

        self._wigner_real = np.zeros((7,7,7), dtype=ctypes.c_void_p)
        self._wigner_imag = np.zeros((7,7,7), dtype=ctypes.c_void_p)

        self._wigner_b_real = np.zeros((7,7,7), dtype=ctypes.c_void_p)
        self._wigner_b_imag = np.zeros((7,7,7), dtype=ctypes.c_void_p)

        self._dev_matrices = []
        self._dev_pointers = []

        def ffs_numpy_swap_memory(arr):
            out = np.zeros_like(arr)
            for ix in range(arr.shape[0]):
                for iy in range(arr.shape[1]):
                    out[ix, iy] = arr[iy, ix]
            return out


        # convert host rotation matrices to device matrices
        for iz, pz in enumerate(range(-3, 4)):
            for iy, py in enumerate(range(-3, 4)):
                for ix, px in enumerate(range(-3, 4)):

                    # forward real
                    f = wigner_f[(pz, py, px)]

                    pa = np.zeros(nlevel, dtype=ctypes.c_void_p)

                    for p in range(nlevel):
                        # forward real

                        o = ffs_numpy_swap_memory(f['real'][p])
                        nn = cuda_base.gpuarray.to_gpu(o)
                        self._dev_matrices.append(nn)
                        pa[p] = self._dev_matrices[-1].ptr

                    # need array of pointers on gpu
                    pa = cuda_base.gpuarray.to_gpu(pa)
                    self._dev_pointers.append(pa)

                    # forward real
                    self._wigner_real[iz, iy, ix] = self._dev_pointers[-1].ptr

                    pa = np.zeros(nlevel, dtype=ctypes.c_void_p)

                    for p in range(nlevel):

                        # forward imag

                        o = ffs_numpy_swap_memory(f['imag'][p])
                        nn = cuda_base.gpuarray.to_gpu(o)
                        self._dev_matrices.append(nn)
                        pa[p] = self._dev_matrices[-1].ptr

                    # need array of pointers on gpu
                    pa = cuda_base.gpuarray.to_gpu(pa)
                    self._dev_pointers.append(pa)

                    # forward imag
                    self._wigner_imag[iz, iy, ix] = self._dev_pointers[-1].ptr


                    f = None
                    # backward real
                    b = wigner_b[(pz, py, px)]

                    pa = np.zeros(nlevel, dtype=ctypes.c_void_p)

                    for p in range(nlevel):
                        # backward real

                        o = ffs_numpy_swap_memory(b['real'][p])
                        nn = cuda_base.gpuarray.to_gpu(o)
                        self._dev_matrices.append(nn)
                        pa[p] = self._dev_matrices[-1].ptr

                    # need array of pointers on gpu
                    pa = cuda_base.gpuarray.to_gpu(pa)
                    self._dev_pointers.append(pa)

                    # backward real
                    self._wigner_b_real[iz, iy, ix] =self._dev_pointers[-1].ptr

                    pa = np.zeros(nlevel, dtype=ctypes.c_void_p)

                    for p in range(nlevel):

                        # backward imag
                        o = ffs_numpy_swap_memory(b['imag'][p])
                        nn = cuda_base.gpuarray.to_gpu(o)
                        self._dev_matrices.append(nn)
                        pa[p] = self._dev_matrices[-1].ptr

                    # need array of pointers on gpu
                    pa = cuda_base.gpuarray.to_gpu(pa)
                    self._dev_pointers.append(pa)

                    # backward imag
                    self._wigner_b_imag[iz, iy, ix] =self._dev_pointers[-1].ptr


        # pointers to pointers on device
        self._wigner_real   = cuda_base.gpuarray.to_gpu(self._wigner_real)
        self._wigner_imag   = cuda_base.gpuarray.to_gpu(self._wigner_imag)
        self._wigner_b_real = cuda_base.gpuarray.to_gpu(self._wigner_b_real)
        self._wigner_b_imag = cuda_base.gpuarray.to_gpu(self._wigner_b_imag)

        self._arn0 = cuda_base.gpuarray.to_gpu(arn0)


        # load multipole to local lib
        with open(str(_SRC_DIR) + \
                          '/FMMSource/CudaTranslateMTLZ.cu') as fh:
            cpp = fh.read()
        with open(str(_SRC_DIR) + \
                          '/FMMSource/CudaTranslateMTLZ.h') as fh:
            hpp = fh.read()
        self._translate_mtl_lib = cuda_build.simple_lib_creator(hpp, cpp,
            'fmm_translate_mtl')

        self.timer_mtl = opt.Timer(runtime.TIMER)

        self._lock = Lock()


    def translate_mtl_pre(self, level, host_halo_tree):
        self.tree_halo[level] = host_halo_tree

    def translate_mtl_async_func(self, level, radius):
        self._translate_mtl(level, radius)

    def translate_mtl_post(self, level, host_plain_tree=None):
        if host_plain_tree is not None:
            self.tree_plain.get(level, host_plain_tree)
            return None

        return self.tree_plain[level]


    def translate_mtl_cart(self, host_halo_tree, level, radius,
                      host_plain_tree=None):

        self.tree_halo[level] = host_halo_tree
        self._translate_mtlz(level, radius)
        if host_plain_tree is not None:
            self.tree_plain.get(level, host_plain_tree)
            return None
        return self.tree_plain[level]

    # perform all mtl stages in one step
    def translate_mtl(self, host_halo_tree, level, radius,
                      host_plain_tree=None):

        self.tree_halo[level] = host_halo_tree
        self._translate_mtlz(level, radius)
        if host_plain_tree is not None:
            self.tree_plain.get(level, host_plain_tree)
            return None
        return self.tree_plain[level]

    def _translate_mtl(self, level, radius):
        self._lock.acquire(True)

        self.timer_mtl.start()
        err = self._translate_mtl_lib['translate_mtl'](
            _check_dtype(self.tree[level].local_grid_cube_size, INT64),
            self.tree_halo.device_pointer(level),
            self.tree_plain.device_pointer(level),
            _check_dtype(self._d_e, REAL),
            _check_dtype(self._d_p, REAL),
            _check_dtype(self._d_a, REAL),
            _check_dtype(self._d_ar, REAL),
            REAL(radius),
            INT64(self.L),
            _check_dtype(self._int_list[level], INT64),
            _check_dtype(self._d_int_tlookup, INT64),
            _check_dtype(self._d_int_plookup, INT64),
            _check_dtype(self._d_int_radius, ctypes.c_double),
            _check_dtype(self._jlookup, INT64),
            _check_dtype(self._klookup, INT64),
            _check_dtype(self._ipower_mtl, REAL),
            INT64(128),
            INT64(cuda_runtime.DEVICE_NUMBER)
        )

        self.timer_mtl.pause()
        self._lock.release()

        cuda_runtime.cuda_err_check(err)

        if err < 0:
            raise RuntimeError("Negative error code caught: {}".format(err))



    def translate_mtlz(self, host_halo_tree, level, radius,
                      host_plain_tree=None):

        self.tree_halo[level] = host_halo_tree
        self._translate_mtlz(level, radius)
        if host_plain_tree is not None:
            self.tree_plain.get(level, host_plain_tree)
            return None
        return self.tree_plain[level]

    def _translate_mtlz(self, level, radius):
        self._lock.acquire(True)
        
        nthreads_per_block = int(ceil((self.L**2)/32))*32
        if nthreads_per_block <= 0 or nthreads_per_block > 1024:
            raise RuntimeError("bad thread count {} for L={}".format(
                nthreads_per_block, self.L))

        self.timer_mtl.start()
        err = self._translate_mtl_lib['translate_mtl_z'](
            _check_dtype(self.tree[level].local_grid_cube_size, INT64),
            self.tree_halo.device_pointer(level),
            self.tree_plain.device_pointer(level),
            self.tmp_plain0.device_pointer(level),
            self.tmp_plain1.device_pointer(level),
            _check_dtype(self._wigner_real  , ctypes.c_void_p),
            _check_dtype(self._wigner_imag  , ctypes.c_void_p),
            _check_dtype(self._wigner_b_real, ctypes.c_void_p),
            _check_dtype(self._wigner_b_imag, ctypes.c_void_p),
            _check_dtype(self._d_a, REAL),
            _check_dtype(self._arn0, REAL),
            REAL(radius),
            INT64(self.L),
            _check_dtype(self._int_list[level], INT64),
            _check_dtype(self._d_int_tlookup, INT64),
            _check_dtype(self._d_int_radius, ctypes.c_double),
            _check_dtype(self._jlookup, INT64),
            _check_dtype(self._klookup, INT64),
            _check_dtype(self._ipower_mtl, REAL),
            INT64(nthreads_per_block),
            INT64(cuda_runtime.DEVICE_NUMBER)
        )

        self.timer_mtl.pause()
        self._lock.release()

        cuda_runtime.cuda_err_check(err)

        if err < 0:
            raise RuntimeError("Negative error code caught: {}".format(err))




























class CudaFMMLocal(object):
    """
    Class to perform local part of fmm
    """
    def __init__(self, width, domain, entry_data, entry_map, free_space,
            dtype, force_unit, energy_unit):

        self.width = width
        self.domain = domain
        self.entry_data = entry_data
        self.entry_map = entry_map
        self.free_space = free_space
        self.dtype = dtype

        self.sh = pairloop.state_handler.StateHandler(state=None, shell_cutoff=width)

        with open(str(_SRC_DIR) + \
                          '/FMMSource/CudaLocalCells.cu') as fh:
            cpp = fh.read()
        with open(str(_SRC_DIR) + \
                          '/FMMSource/CudaLocalCells.h') as fh:
            hpp = fh.read()

        hpp = hpp % {
            'SUB_FORCE_UNIT': str(force_unit),
            'SUB_ENERGY_UNIT': str(energy_unit)
        }

        self._lib = cuda_build.simple_lib_creator(hpp, cpp, 'fmm_local')
        self._lib0 = self._lib['local_cell_by_cell_0']
        self._lib1 = self._lib['local_cell_by_cell_1']
        self._lib2 = self._lib['local_cell_by_cell_2']

        #print("CUDA LOCAL BUILT")
        self._global_size = np.zeros(3, dtype=INT64)
        self._global_size[:] = entry_map.cube_side_count

        self._ncells =  (self._global_size[0] + 6) * \
                        (self._global_size[1] + 6) * \
                        (self._global_size[2] + 6)

        self._local_size = np.zeros(3, dtype=INT64)
        self._local_size[:] = self.entry_data.local_size[:]
        
        self._local_offset = np.zeros(3, dtype=INT64)
        self._local_offset[:] = self.entry_data.local_offset[:]
        self._u = np.zeros(1, dtype=self.dtype)
        self.last_u = 0.0

        self._ll_array = np.zeros(1, dtype=INT64)
        self._ll_ccc_array = np.zeros(self._ncells, dtype=INT64)
        self.d_ll_ccc_array = cuda_base.gpuarray.GPUArray(shape=self._ll_ccc_array.shape, dtype=INT64)

        self._ntotal = 100000
        self.d_positions = cuda_base.gpuarray.GPUArray(shape=(self._ntotal, 3), dtype=REAL)
        self.d_charges = cuda_base.gpuarray.GPUArray(shape=(self._ntotal, 1), dtype=REAL)
        self.d_forces = cuda_base.gpuarray.GPUArray(shape=(self._ntotal, 3), dtype=REAL)
        self.d_potential_array = cuda_base.gpuarray.GPUArray(shape=(self._ntotal, 1), dtype=REAL)

        self.h_forces = np.zeros(shape=(self._ntotal, 3), dtype=REAL)
        self.h_potential_array = np.zeros(shape=(self._ntotal, 1), dtype=REAL)

        self.exec_count = 0


        self.timer0 = opt.Timer(runtime.TIMER)
        self.timer1 = opt.Timer(runtime.TIMER)
        self.timer2 = opt.Timer(runtime.TIMER)

    def __call__(self, positions, charges, forces, cells, potential=None):
        dats = {
            'p': positions(READ),
            'q': charges(READ),
            'f': forces(INC),
            'c': cells(READ)
        }
        
        self._dats = dats

        if potential is not None and \
                issubclass(type(potential), ParticleDat):
            dats['u'] = potential(INC_ZERO)
            assert potential[:].shape[0] >= positions.npart_local
        elif potential is not None:
            assert potential.shape[0] * potential.shape[1] >= \
                    postitions.npart_local

        self._u[0] = 0.0

        nlocal, nhalo, ncell = self.sh.pre_execute(dats=dats)
        self._tmp_dict = {
            'nlocal': nlocal,
            'nhalo': nhalo,
            'ncell': ncell
        }
        ntotal = nlocal + nhalo

        if self._ll_array.shape[0] < (ntotal + self._ncells):
            self._ll_array = np.zeros(ntotal+100+self._ncells, dtype=INT64)
            self.d_ll_array = cuda_base.gpuarray.GPUArray(shape=self._ll_array.shape, dtype=INT64)
        
        
        if ntotal > self._ntotal:
            self.d_positions = cuda_base.gpuarray.GPUArray(shape=(ntotal, 3), dtype=REAL)
            self.d_charges = cuda_base.gpuarray.GPUArray(shape=(ntotal, 1), dtype=REAL)
            self.d_forces = cuda_base.gpuarray.GPUArray(shape=(ntotal, 3), dtype=REAL)
            self.d_potential_array = cuda_base.gpuarray.GPUArray(shape=(ntotal, 1), dtype=REAL)
            self.h_forces = np.zeros(shape=(ntotal, 3), dtype=REAL)
            self.h_potential_array = np.zeros(shape=(ntotal, 1), dtype=REAL)           
            self._ntotal = ntotal
    
    def call2(self,  positions, charges, forces, cells, potential=None):
        nlocal = self._tmp_dict['nlocal']
        nhalo = self._tmp_dict['nhalo']
        ncell = self._tmp_dict['ncell']

        ntotal = nlocal + nhalo
        compute_pot = INT64(0)
        dummy_real = REAL(0)
        pot_ptr = ctypes.byref(dummy_real)
        if potential is not None:
            compute_pot.value = 1
            # pot_ptr = _check_dtype(potential, REAL)
            pot_ptr = potential.ctypes_data

        if self.domain.extent.dtype is not REAL:
            raise RuntimeError("expected c_double extent")
        
        if self.free_space == '27':
            free_space = 0
        elif self.free_space == True:
            free_space = 1
        else:
            free_space = 0

        exec_count = INT64(0)
        req_len = INT64(0)
        ret_max_cc = INT64(0)
        self.timer0.start()
        err = self._lib0(
            INT64(free_space),
            self.domain.extent.ctypes_data,
            self._global_size.ctypes.get_as_parameter(),
            self._local_size.ctypes.get_as_parameter(),
            self._local_offset.ctypes.get_as_parameter(),
            INT64(runtime.NUM_THREADS),
            INT64(nlocal),
            INT64(ntotal),
            self.sh.get_pointer(positions(READ)),
            self.sh.get_pointer(charges(READ)),
            self.sh.get_pointer(cells(READ)),
            self._ll_array.ctypes.get_as_parameter(),
            self._ll_ccc_array.ctypes.get_as_parameter(),
            INT64(512),
            INT64(cuda_runtime.DEVICE_NUMBER),
            _check_dtype(self.d_positions, REAL),
            _check_dtype(self.d_charges, REAL),
            _check_dtype(self.d_forces, REAL),
            _check_dtype(self.d_potential_array, REAL),
            ctypes.byref(req_len),
            ctypes.byref(ret_max_cc)
        )

        cuda_runtime.cuda_err_check(err)
        self.timer0.pause()

        
        self.timer1.start()
        err = self._lib1(
            INT64(free_space),
            self._global_size.ctypes.get_as_parameter(),
            self._local_size.ctypes.get_as_parameter(),
            self._local_offset.ctypes.get_as_parameter(),
            INT64(runtime.NUM_THREADS),
            INT64(nlocal),
            INT64(ntotal),
            ctypes.byref(exec_count),
            self._ll_array.ctypes.get_as_parameter(),
            self._ll_ccc_array.ctypes.get_as_parameter(),
            INT64(256),
            INT64(cuda_runtime.DEVICE_NUMBER),
            _check_dtype(self.d_positions, REAL),
            _check_dtype(self.d_charges, REAL),
            _check_dtype(self.d_forces, REAL),
            _check_dtype(self.d_potential_array, REAL),
            _check_dtype(self.d_ll_array, INT64),
            _check_dtype(self.d_ll_ccc_array, INT64),
            ret_max_cc,
            _check_dtype(self.h_forces, REAL),
            _check_dtype(self.h_potential_array, REAL)
        )

        self.timer1.pause()
        cuda_runtime.cuda_err_check(err)

        err = self._lib2(
            compute_pot,
            INT64(nlocal),
            INT64(cuda_runtime.DEVICE_NUMBER),
            _check_dtype(self.h_forces, REAL),
            _check_dtype(self.h_potential_array, REAL),
            _check_dtype(forces, REAL),
            pot_ptr,
            _check_dtype(self._u, REAL)
        )

        cuda_runtime.cuda_err_check(err)


        self.exec_count += exec_count.value
        self.last_u = self._u[0]
        return self._u[0]

    def call3(self):
        self.timer2.start()
        self.sh.post_execute(dats=self._dats)
        self.timer2.pause()
        
        self._update_opt()

    
    def _update_opt(self):
        p = opt.PROFILE
        b = self.__class__.__name__ + ':'
        p[b+'init'] = self.timer0.time()
        p[b+'local'] = self.timer1.time()
        p[b+'finalise'] = self.timer2.time()








