"""
Module to hold CUDA related code.
"""

import os
import ctypes as ct
import math
import build
import subprocess
import hashlib
import atexit
import runtime
import pio
import kernel
import data
import access
import cell
import mpi
import host

ERROR_LEVEL = runtime.Level(2)


#####################################################################################
# NVCC Compiler
##################################################################################### '--maxrregcount=40'

# '--ptxas-options=-v -dlcm=cg'

NVCC = build.Compiler(['nvcc_system_default'],
                      ['nvcc'],
                      ['-Xcompiler', '"-fPIC"'],
                      ['-lm'],
                      ['-O3', '--ptxas-options=-v -dlcm=ca', '--maxrregcount=64'], # '-O3', '-Xptxas', '"-v"', '-lineinfo'
                      ['-G', '-g', '--source-in-ptx', '--ptxas-options="-v -dlcm=ca"'],
                      ['-c', '-arch=sm_35', '-m64', '-lineinfo'],
                      ['-shared', '-Xcompiler', '"-fPIC"'],
                      '__restrict__')

#####################################################################################
# build static libs
#####################################################################################


def _build_static_libs(lib):

    with open(runtime.LIB_DIR.dir + lib + ".cu", "r") as fh:
        _code = fh.read()
        fh.close()
    with open(runtime.LIB_DIR.dir + lib + ".h", "r") as fh:
        _code += fh.read()
        fh.close()

    _m = hashlib.md5()
    _m.update(_code)
    _m = _m.hexdigest()

    _lib_filename = os.path.join(runtime.BUILD_DIR.dir, lib + '_' +str(_m) +'.so')

    if mpi.MPI_HANDLE.rank == 0:
        if not os.path.exists(_lib_filename):


            _lib_src_filename = runtime.LIB_DIR.dir + lib + '.cu'

            _c_cmd = NVCC.binary + [_lib_src_filename] + ['-o'] + [_lib_filename] + NVCC.c_flags + NVCC.l_flags
            if runtime.DEBUG.level > 0:
                _c_cmd += NVCC.dbg_flags
            else:
                _c_cmd += NVCC.opt_flags

            _c_cmd += NVCC.shared_lib_flag

            if runtime.VERBOSE.level > 2:
                print "Building", _lib_filename

            stdout_filename = runtime.BUILD_DIR.dir + lib + '_' +str(_m) + '.log'
            stderr_filename = runtime.BUILD_DIR.dir + lib + '_' +str(_m) + '.err'
            try:
                with open(stdout_filename, 'w') as stdout:
                    with open(stderr_filename, 'w') as stderr:
                        stdout.write('Compilation command:\n')
                        stdout.write(' '.join(_c_cmd))
                        stdout.write('\n\n')
                        p = subprocess.Popen(_c_cmd,
                                             stdout=stdout,
                                             stderr=stderr)
                        p.communicate()
            except:
                if ERROR_LEVEL.level > 2:
                    raise RuntimeError('gpucuda error: helper library not built.')
                elif runtime.VERBOSE.level > 2:
                    print "gpucuda warning: Shared library not built:", lib



    mpi.MPI_HANDLE.barrier()

    return _lib_filename


#####################################################################################
# load libraries
#####################################################################################

try:
    CUDA_INC_PATH = os.environ['CUDA_INSTALL_PATH']
except KeyError:
    if ERROR_LEVEL.level > 2:
        raise RuntimeError('gpucuda error: cuda toolkit environment path not '
                           'found, expecting CUDA_INSTALL_PATH')
    CUDA_INC_PATH = None

try:
    LIBCUDART = ct.cdll.LoadLibrary(CUDA_INC_PATH + "/lib64/libcudart.so.6.5")

except:
    if ERROR_LEVEL.level > 2:
        raise RuntimeError('gpucuda error: Module is not initialised correctly,'
                           ' CUDA runtime not loaded')
    LIBCUDART = None

# wrapper library for functions involving types.
try:
    LIBHELPER = ct.cdll.LoadLibrary(_build_static_libs('cudaHelperLib'))
except:
    LIBHELPER = None


#####################################################################################
# Device id of currently used device. Assuming model of one mpi process per gpu.
#####################################################################################

class Device(object):
    def __init__(self):
        self._dev_id = None

    @property
    def id(self):
        return self._dev_id

    @id.setter
    def id(self, new_id):
        self._dev_id = int(new_id)

DEVICE = Device()

#####################################################################################
# cuda_set_device. Assuming model of one mpi process per gpu.
#####################################################################################

def cuda_set_device(device=None):
    """
    Set the cuda device.
    :param int device: Dev id to use. If not set a device will be chosen based on rank.
    :return:
    """
    if device is None:
        _r = 0

        try:
            _mv2r = os.environ['MV2_COMM_WORLD_LOCAL_RANK']
        except KeyError:
            _mv2r = None

        try:
            _ompr = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
        except KeyError:
            _ompr = None

        if (_mv2r is None) and (_ompr is None):
            print("gpucuda warning: Did not find local rank, defaulting to device 0")

        if (_mv2r is not None) and (_ompr is not None):
            print("gpucuda warning: Found two local ranks, defaulting to device 0")

        if _mv2r is not None:
            _r = int(_mv2r) % mpi.MPI_HANDLE.nproc
        if _ompr is not None:
            _r = int(_ompr) % mpi.MPI_HANDLE.nproc

    else:
        _r = int(device)

    if LIBCUDART is not None:
        if runtime.VERBOSE.level > 0:
            pio.rprint("setting device ", _r)

        LIBCUDART['cudaSetDevice'](ct.c_int(_r))
        libcudart('cudaSetDeviceFlags',ct.c_uint(8))
        DEVICE.id = _r
    else:
        pio.rprint("gpucuda warning: No device set")

    if (runtime.VERBOSE.level > 0) and (LIBCUDART is not None):
        _dev = ct.c_int()
        LIBCUDART['cudaGetDevice'](ct.byref(_dev))
        pio.rprint("cudaGetDevice returned device ", _dev.value)

#####################################################################################
# Is module ready to use?
#####################################################################################

def INIT_STATUS():
    """
    Function to determine if the module is correctly loaded and can be used.
    :return: True/False.
    """

    if (LIBCUDART is not None) and (LIBHELPER is not None) and (DEVICE.id is not None) and runtime.CUDA_ENABLED.flag:
        return True
    else:
        return False


#####################################################################################
# cuda_err_checking
#####################################################################################

def cuda_err_check(err_code):
    """
    Wrapper to check cuda error codes.
    :param err_code:
    :return:
    """
    if LIBHELPER is not None:
        LIBHELPER['cudaErrorCheck'](err_code)


def libcudart(*args):
    """
    Wrapper to cuda runtime library with error code checking.
    :param args: string <function name>, args.
    :return:
    """

    assert LIBCUDART is not None, "gpucuda error: No CUDA Runtime library loaded"

    if runtime.VERBOSE.level > 2:
        pio.rprint(args)

    cuda_err_check(LIBCUDART[args[0]](*args[1::]))

#####################################################################################
# cuda_host_register
#####################################################################################

REGISTERED_PTRS = []


def cuda_host_register(dat):
    """
    Page lock memory allocated in dat.
    :param dat: dat containing memory to page lock.
    :return:
    """
    if type(dat) == data.ScalarArray:
        _s = ct.c_size_t(dat.max_size * ct.sizeof(dat.dtype))
    elif type(dat) == data.ParticleDat:
        _s = ct.c_size_t(dat.max_size * dat.ncomp * ct.sizeof(dat.dtype))

    LIBHELPER['cudaHostRegisterWrapper'](dat.ctypes_data, _s)
    REGISTERED_PTRS.append(dat.ctypes_data)


def cuda_host_unregister(dat):
    """
    Page unlock memory allocated in dat.
    :param dat: dat containing memory to page unlock.
    :return:
    """
    if dat.ctypes_data in REGISTERED_PTRS:
        LIBHELPER['cudaHostUnregisterWrapper'](dat.ctypes_data)
        REGISTERED_PTRS.remove(dat.ctypes_data)

def cuda_host_unregister_all():
    """
    Page unlock memory allocated in dat.
    :param dat: dat containing memory to page unlock.
    :return:
    """
    for ctypes_data in REGISTERED_PTRS:
        LIBHELPER['cudaHostUnregisterWrapper'](ctypes_data)
        REGISTERED_PTRS.remove(ctypes_data)


#####################################################################################
# cuda_device_reset
#####################################################################################


def cuda_device_reset():
    """
    Reset the current cuda device.
    :return:
    """

    if (DEVICE.id is not None) and (LIBCUDART is not None):
        libcudart('cudaDeviceReset')

#####################################################################################
# CudaDeviceDat
#####################################################################################


class CudaDeviceDat(object):
    """
    Class to handle memory on cuda devices.
    :arg int size: Number of elements in array.
    :arg dtype: Data type default ctypes.c_double.
    """

    def __init__(self, size=8, dtype=ct.c_double):

        self._size = size
        self._dtype = dtype

        # create device pointer.
        self._d_p = ct.POINTER(self._dtype)()

        # allocate space.
        libcudart('cudaMalloc', ct.byref(self._d_p),
                  ct.c_size_t(self._size))

        # Bool to flag if memory is allocated on device.
        self._alloc = True


    def free(self):
        """
        Free allocated device memory. Do not call after a device reset.
        :return:
        """
        libcudart('cudaFree', self._d_p)


    def cpy_htd(self, host_ptr, size=None):
        """
        Copy data from host pointer to device.
        :param host_ptr: host pointer to copy from.
        :param size: amount to copy (bytes). If not specified Method will copy
        with size equal to the length of the allocated device array.
        :return:
        """


        if size is None:
            _s = ct.c_size_t(self._size)
        else:
            _s = ct.c_size_t(size)

        LIBHELPER['cudaCpyHostToDevice'](self._d_p, host_ptr,_s)

    def cpy_dth(self, host_ptr, size=None):
        """
        Copy data from host pointer to device.
        :param host_ptr: host pointer to copy into.
        :param size: amount to copy (bytes). If not specified Method
        will copy with size equal to the length of the allocated device array.
        :return:
        """

        if size is None:
            _s = ct.c_size_t(self._size)
        else:
            _s = ct.c_size_t(size)

        LIBHELPER['cudaCpyDeviceToHost'](host_ptr, self._d_p, _s)

    @property
    def dtype(self):
        """
        :return: Data type of array.
        """
        return self._dtype

    @property
    def ctypes_data(self):
        """
        Get pointer to device memory.
        :return: device pointer
        """
        return self._d_p

    def resize(self, size):
        """
        Resize the array, will be slow.
        :param size:
        :return:
        """
        _d_tp = ct.POINTER(self._dtype)()
        libcudart('cudaMalloc', ct.byref(_d_tp), ct.c_size_t(size))
        self.free()
        self._d_p = _d_tp

#####################################################################################
# Registered Cuda Dats
#####################################################################################

class _RegisteredDats(object):

    def __init__(self):
        self._reg_dats = dict()

    def register(self, dat):
        self._reg_dats[dat] = CudaDeviceDat(dat.size, dat.dtype)

    def get_cuda_dat(self, dat):
        return self._reg_dats[dat]

    def __getitem__(self, dat):
        return self._reg_dats[dat]


    def get_device_pointer(self, dat):
        return self._reg_dats[dat].ctypes_data

    def copy_to_device(self, dat, ptr=None):
        if ptr is None:
            self._reg_dats[dat].cpy_htd(dat.ctypes_data)
        else:
            self._reg_dats[dat].cpy_htd(ptr)

    def copy_from_device(self, dat, ptr=None):
        if ptr is None:
            self._reg_dats[dat].cpy_dth(dat.ctypes_data)
        else:
            self._reg_dats[dat].cpy_dth(ptr)


CUDA_DATS = _RegisteredDats()



#####################################################################################
# Module Init
#####################################################################################

cuda_set_device()

#####################################################################################
# Module cleanup
#####################################################################################

def gpucuda_cleanup():
    cuda_host_unregister_all()
    cuda_device_reset()


atexit.register(gpucuda_cleanup)


#####################################################################################
# CUDA a=b+c test kernel.
#####################################################################################

class aebpc(object):
    def __init__(self,a,b,c):

        _kernel_code = '''
            a[0] = b[0] + c[0] + d + e;
        '''

        _dat_dict = {
            'a': a,
            'b': b,
            'c': c
        }

        _consts = (kernel.Constant('e', 4),)

        _static_args = {'d': ct.c_int}


        _kernel = kernel.Kernel('aebpc', _kernel_code, _consts, ['stdio.h'], None, _static_args)

        self._lib = SingleAllParticleLoop(None, _kernel, _dat_dict)

    def execute(self):
        self._lib.execute(static_args={'d': ct.c_int(7)})


#####################################################################################
# CUDA _Base
#####################################################################################


class _Base(object):
    def __init__(self, types_map, kernel, particle_dat_dict):
        self._types_map = types_map
        self._kernel = kernel
        self._particle_dat_dict = particle_dat_dict

        # set compiler
        self._cc = NVCC
        self._temp_dir = runtime.BUILD_DIR.dir

        #start code creation
        self._static_arg_init()
        self._code_init()


        # set unique names.
        self._unique_name = 'CUDA_' + self._unique_name_calc()
        self._library_filename = self._unique_name + '.so'

        # see if exists
        if not os.path.exists(os.path.join(self._temp_dir, self._library_filename)):

            if mpi.MPI_HANDLE.rank == 0:
                self._create_library()
            mpi.MPI_HANDLE.barrier()

        self._lib = ct.cdll.LoadLibrary(os.path.join(self._temp_dir, self._library_filename))

    def _unique_name_calc(self):
        """Return name which can be used to identify the pair loop
        in a unique way.
        """
        return self._kernel.name + '_' + self.hexdigest()

    def hexdigest(self):
        """Create unique hex digest"""
        m = hashlib.md5()
        m.update(self._kernel.code + self._code)
        if self._kernel.headers is not None:
            for header in self._kernel.headers:
                m.update(header)
        return m.hexdigest()

    def _included_headers(self):
        """Return names of included header files."""
        s = ''
        if self._kernel.headers is not None:
            s += '\n'
            for x in self._kernel.headers:
                s += '#include \"' + x + '\" \n'
        return s

    def _static_arg_init(self):
        """
        Create code to handle device constants.
        """

        self._device_const_dec = ''
        self._device_const_copy = ''

        if self._kernel.static_args is not None:

            for ix in self._kernel.static_args.items():
                self._device_const_dec += '__constant__ ' + host.ctypes_map[ix[1]] + ' ' + ix[0] + '; \n'
                self._device_const_copy += 'checkCudaErrors(cudaMemcpyToSymbol(' + ix[0] + ', &h_' + ix[0] + ', sizeof(h_' + ix[0] + '))); \n'


    def _argnames(self):
        """
        Comma separated string of argument name declarations.

        This string of argument names is used in the declaration of
        the method which executes the pairloop over the grid.
        If, for example, the pairloop gets passed two particle_dats,
        then the result will be ``double** arg_000,double** arg_001`.`
        """

        self._argtypes = []

        argnames = ''
        if self._kernel.static_args is not None:
            self._static_arg_order = []

            for i, dat in enumerate(self._kernel.static_args.items()):
                argnames += 'const ' + host.ctypes_map[dat[1]] + ' h_' + dat[0] + ','
                self._static_arg_order.append(dat[0])
                self._argtypes.append(dat[1])

        for i, dat_orig in enumerate(self._particle_dat_dict.items()):

            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
                _mode = dat_orig[1][1]
            else:
                dat = dat_orig
                _mode = access.RW

            argnames += self._mode_arg_dec_str(_mode) + host.ctypes_map[dat[1].dtype] + ' * ' + self._cc.restrict_keyword + ' h_' + dat[0] + ','
            self._argtypes.append(dat[1].dtype)

        return argnames[:-1]


    def _kernel_argnames(self):
        """
        Comma separated string of argument names to be passed to kernel launch.
        """
        args = ''
        for arg in self._particle_dat_dict.items():
            args += 'h_' + arg[0] + ','
        return args[:-1]

    def _mode_arg_dec_str(self, mode):
        _s = ' '
        if mode is access.R:
            _s = 'const '

        return _s




    def _kernel_argument_declarations(self):
        """Define and declare the kernel arguments.

        For each argument the kernel gets passed a pointer of type
        ``double* loc_argXXX[2]``. Here ``loc_arg[i]`` with i=0,1 is
        pointer to the data which contains the properties of particle i.
        These properties are stored consecutively in memory, so for a
        scalar property only ``loc_argXXX[i][0]`` is used, but for a vector
        property the vector entry j of particle i is accessed as
        ``loc_argXXX[i][j]``.

        This method generates the definitions of the ``loc_argXXX`` variables
        and populates the data to ensure that ``loc_argXXX[i]`` points to
        the correct address in the particle_dats.
        """
        s = ''

        for i, dat_orig in enumerate(self._particle_dat_dict.items()):
            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
                _mode = dat_orig[1][1]
            else:
                dat = dat_orig
                _mode = access.RW
            loc_argname = dat[0]

            if type(dat[1]) == data.ParticleDat:
                s += self._mode_arg_dec_str(_mode) + host.ctypes_map[dat[1].dtype] + ' * ' + self._cc.restrict_keyword + ' d_' + loc_argname + ','
            if type(dat[1]) == data.ScalarArray:
                s += self._mode_arg_dec_str(_mode) + host.ctypes_map[dat[1].dtype] + ' * ' + self._cc.restrict_keyword + ' d_' + loc_argname + ','
        return s[:-1]

    def _kernel_pointer_mapping(self):
        """
        Create string for thread id and pointer mapping.
        """
        _s = ''

        space = ' ' * 14

        for dat_orig in self._particle_dat_dict.items():

            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
                _mode = dat_orig[1][1]
            else:
                dat = dat_orig
                _mode = access.RW

            argname = 'd_' + dat[0]
            loc_argname = dat[0]

            if type(dat[1]) == data.ParticleDat:
                _s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ';\n'
                _s += space + loc_argname + ' = ' + argname + '+' + str(dat[1].ncomp) + '*_ix;\n'

        return _s

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H
        #include "%(LIB_DIR)s/cuda_generic.h"

        %(INCLUDED_HEADERS)s

        extern "C" void %(KERNEL_NAME)s_wrapper(const int blocksize[3], const int threadsize[3], const int _h_n, %(ARGUMENTS)s);

        #endif
        '''

        d = {'UNIQUENAME': self._unique_name,
             'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir}
        return code % d


    def _code_init(self):
        self._kernel_code = self._kernel.code

        self._code = '''
        #include \"%(UNIQUENAME)s.h\"

        //device constant decelerations.
        __constant__ int _d_n;
        %(DEVICE_CONSTANT_DECELERATION)s

        //device kernel decelerations.
        __global__ void %(KERNEL_NAME)s_gpukernel(%(KERNEL_ARGUMENTS_DECL)s){

            int _ix = threadIdx.x + blockIdx.x*blockDim.x;
            if (_ix < _d_n){
            %(GPU_POINTER_MAPPING)s

                %(GPU_KERNEL)s
            }
        }

        void %(KERNEL_NAME)s_wrapper(const int blocksize[3], const int threadsize[3], const int _h_n, %(ARGUMENTS)s) {

            //device constant copy.
            checkCudaErrors(cudaMemcpyToSymbol(_d_n, &_h_n, sizeof(_h_n)));
            %(DEVICE_CONSTANT_COPY)s

            dim3 bs; bs.x = blocksize[0]; bs.y = blocksize[1]; bs.z = blocksize[2];
            dim3 ts; ts.x = threadsize[0]; ts.y = threadsize[1]; ts.z = threadsize[2];


            %(KERNEL_NAME)s_gpukernel<<<bs,ts>>>(%(KERNEL_ARGUMENTS)s);
            checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError(" %(KERNEL_NAME)s Execution failed. \\n");
        }
        '''

    def _generate_impl_source(self):
        """Generate the source code the actual implementation.
        """

        d = {'UNIQUENAME': self._unique_name,
             'GPU_POINTER_MAPPING': self._kernel_pointer_mapping(),
             'GPU_KERNEL': self._kernel_code,
             'ARGUMENTS': self._argnames(),
             'KERNEL_ARGUMENTS': self._kernel_argnames(),
             'KERNEL_NAME': self._kernel.name,
             'KERNEL_ARGUMENTS_DECL': self._kernel_argument_declarations(),
             'DEVICE_CONSTANT_DECELERATION': self._device_const_dec,
             'DEVICE_CONSTANT_COPY': self._device_const_copy}

        return self._code % d

    def _create_library(self):
            """
            Create a shared library from the source code.
            """
            filename_base = os.path.join(self._temp_dir, self._unique_name)

            header_filename = filename_base + '.h'
            impl_filename = filename_base + '.cu'
            with open(header_filename, 'w') as f:
                print >> f, self._generate_header_source()
            with open(impl_filename, 'w') as f:
                print >> f, self._generate_impl_source()

            object_filename = filename_base + '.o'
            library_filename = filename_base + '.so'

            if runtime.VERBOSE.level > 2:
                print "Building", library_filename

            cflags = []
            cflags += self._cc.c_flags

            if runtime.DEBUG.level > 0:
                cflags += self._cc.dbg_flags
            else:
                cflags += self._cc.opt_flags


            cc = self._cc.binary
            ld = self._cc.binary
            lflags = self._cc.l_flags

            compile_cmd = cc + self._cc.compile_flag + cflags + ['-I', self._temp_dir] + ['-o', object_filename,
                                                                                          impl_filename]

            link_cmd = ld + self._cc.shared_lib_flag + lflags + ['-o', library_filename, object_filename]
            stdout_filename = filename_base + '.log'
            stderr_filename = filename_base + '.err'
            with open(stdout_filename, 'w') as stdout:
                with open(stderr_filename, 'w') as stderr:
                    stdout.write('Compilation command:\n')
                    stdout.write(' '.join(compile_cmd))
                    stdout.write('\n\n')
                    p = subprocess.Popen(compile_cmd,
                                         stdout=stdout,
                                         stderr=stderr)
                    p.communicate()
                    stdout.write('Link command:\n')
                    stdout.write(' '.join(link_cmd))
                    stdout.write('\n\n')
                    p = subprocess.Popen(link_cmd,
                                         stdout=stdout,
                                         stderr=stderr)
                    p.communicate()

    def execute(self, dat_dict=None, static_args=None):

        """Allow alternative pointers"""
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict

        if self._particle_dat_dict.values()[0].npart < 1025:
            _blocksize = (ct.c_int * 3)(1, 1, 1)
            _threadsize = (ct.c_int * 3)(self._particle_dat_dict.values()[0].npart, 1, 1)

        else:
            _blocksize = (ct.c_int * 3)(int(math.ceil(self._particle_dat_dict.values()[0].npart / 1024.)), 1, 1)
            _threadsize = (ct.c_int * 3)(1024, 1, 1)


        args = [_blocksize, _threadsize, ct.c_int(self._particle_dat_dict.values()[0].npart)]

        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Add pointer arguments to launch command'''
        for dat in self._particle_dat_dict.values():
            if type(dat) is tuple:
                args.append(dat[0].get_cuda_dat().ctypes_data)
            else:
                args.append(dat.get_cuda_dat().ctypes_data)

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        method(*args)

#####################################################################################
# CUDA SingleAllParticleLoop
#####################################################################################

class SingleAllParticleLoop(_Base):
    pass

#####################################################################################
# CUDA SingleParticleLoop
#####################################################################################

class SingleParticleLoop(_Base):
    pass

# TODO: Add start and end points to this.

#####################################################################################
# CUDA SimpleCudaPairLoop
#####################################################################################

class SimpleCudaPairLoop(_Base):
    def __init__(self, n, domain, potential, dat_dict):
        self._N = n
        self._domain = domain
        self._potential = potential
        self._particle_dat_dict = dat_dict


        self._cc = NVCC
        self._temp_dir = runtime.BUILD_DIR.dir
        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)
        self._kernel = self._potential.kernel

        self._nargs = len(self._particle_dat_dict)

        self._static_arg_init()
        self._code_init()

        self._unique_name = 'CUDA_' + self._unique_name_calc()

        self._library_filename = self._unique_name + '.so'

        if not os.path.exists(os.path.join(self._temp_dir, self._library_filename)):
            if mpi.MPI_HANDLE is None:
                self._create_library()
            else:
                if mpi.MPI_HANDLE.rank == 0:
                    self._create_library()
                mpi.MPI_HANDLE.barrier()

        try:
            self._lib = ct.cdll.LoadLibrary(os.path.join(self._temp_dir, self._library_filename))
        except OSError as e:
            raise OSError(e)
        except:
            build.load_library_exception(self._kernel.name, self._unique_name, type(self))

    def hexdigest(self):
        """Create unique hex digest"""

        m = hashlib.md5()
        m.update(self._kernel.code + self._code + str(self._domain.extent))
        if self._kernel.headers is not None:
            for header in self._kernel.headers:
                m.update(header)
        return m.hexdigest()

    def _code_init(self):
        self._kernel_code = self._kernel.code

        self._code = '''
        #include \"%(UNIQUENAME)s.h\"

        //device constant decelerations.
        __constant__ int _d_n;
        __constant__ int _d_cell_offset;
        __constant__ int _d_cell_array[3];
        __constant__ double _d_extent[3];

        __constant__ int cell_map[27][3] = {
                                        {-1,1,-1},
                                        {-1,-1,-1},
                                        {-1,0,-1},
                                        {0,1,-1},
                                        {0,-1,-1},
                                        {0,0,-1},
                                        {1,0,-1},
                                        {1,1,-1},
                                        {1,-1,-1},

                                        {-1,1,0},
                                        {-1,0,0},
                                        {-1,-1,0},
                                        {0,-1,0},
                                        {0,0,0},
                                        {0,1,0},
                                        {1,0,0},
                                        {1,1,0},
                                        {1,-1,0},

                                        {-1,0,1},
                                        {-1,1,1},
                                        {-1,-1,1},
                                        {0,0,1},
                                        {0,1,1},
                                        {0,-1,1},
                                        {1,0,1},
                                        {1,1,1},
                                        {1,-1,1}
                                    };

        %(DEVICE_CONSTANT_DECELERATION)s


        __device__ void cell_index_offset(int cp, int cpp_i, int * cpp, int *flag, double *offset){

            int tmp = _d_cell_array[0]*_d_cell_array[1];
            int Cz = cp/(_d_cell_array[0]*_d_cell_array[1]);
            int Cx = cp %% _d_cell_array[0];
            int Cy = (cp - Cz*tmp)/_d_cell_array[0];

            Cx += cell_map[cpp_i][0];
            Cy += cell_map[cpp_i][1];
            Cz += cell_map[cpp_i][2];

            int C0 = (Cx + _d_cell_array[0]) %% _d_cell_array[0];
            int C1 = (Cy + _d_cell_array[1]) %% _d_cell_array[1];
            int C2 = (Cz + _d_cell_array[2]) %% _d_cell_array[2];

            if ((Cx != C0) || (Cy != C1) || (Cz != C2)) {
                *flag = 1;
                offset[0] = ((double)sign(Cx - C0))*_d_extent[0];
                offset[1] = ((double)sign(Cy - C1))*_d_extent[1];
                offset[2] = ((double)sign(Cz - C2))*_d_extent[2];

            } else {*flag = 0; }

            *cpp = (C2*_d_cell_array[1] + C1)*_d_cell_array[0] + C0;

            return;
        }

        //device kernel decelerations.
        __global__ void %(KERNEL_NAME)s_gpukernel(const int * __restrict__ CCC, const int * __restrict__ PCL, const int * __restrict__ cell_list, %(KERNEL_ARGUMENTS_DECL)s){

            int _ix = threadIdx.x + blockIdx.x*blockDim.x;
            if (_ix < _d_n){

                //create local store for acceleration of particle _ix.
                double _a[3];
                _a[0] = 0; _a[1] = 0; _a[2] = 0;

                //ensure positon of particle _ix is only read once.
                 const double _p[3] = {%(POS_VECTOR)s[_ix*3],
                                        %(POS_VECTOR)s[_ix*3 + 1],
                                        %(POS_VECTOR)s[_ix*3 + 2]};

                for(int cpp_i=0; cpp_i<27; cpp_i++){
                    double s[3];
                    int flag, cpp;

                    cell_index_offset(PCL[_ix], cpp_i, &cpp, &flag, s);

                    if (cell_list[_d_cell_offset+cpp] > -1){
                        for(int _iy = cell_list[_d_cell_offset+cpp];
                         _iy < cell_list[_d_cell_offset+cpp]+CCC[cpp];
                         _iy++){

                            if (_iy != _ix){

                            double r1[3];
                            %(GPU_POINTER_MAPPING)s

                            const double R0 = P[1][0] - P[0][0];
                            const double R1 = P[1][1] - P[0][1];
                            const double R2 = P[1][2] - P[0][2];

                            const double r2 = R0*R0 + R1*R1 + R2*R2;

                            if (r2 < 6.25){

                                const double r_m2 = 1.0/r2;
                                const double r_m4 = r_m2*r_m2;
                                const double r_m6 = r_m4*r_m2;

                                //u[0]+= 4.0*((r_m6-1.0)*r_m6 + 0.004079222784);

                                const double r_m8 = r_m4*r_m4;
                                const double f_tmp = -48.0*(r_m6 - 0.5)*r_m8;

                                A[0][0]+=f_tmp*R0;
                                A[0][1]+=f_tmp*R1;
                                A[0][2]+=f_tmp*R2;

                                A[1][0]-=f_tmp*R0;
                                A[1][1]-=f_tmp*R1;
                                A[1][2]-=f_tmp*R2;

                            }



                            /*
                            %(GPU_KERNEL)s
                            */

                            }

                        }
                    }
                }

                //Write acceleration to dat.
                %(ACCEL_VECTOR)s[_ix*3]     = _a[0];
                %(ACCEL_VECTOR)s[_ix*3 + 1] = _a[1];
                %(ACCEL_VECTOR)s[_ix*3 + 2] = _a[2];

            }
            return;
        }

        void %(KERNEL_NAME)s_wrapper(const int blocksize[3],
                                     const int threadsize[3],
                                     const int _h_n,
                                     const int _h_cell_offset,
                                     const int _h_cell_array[3],
                                     const double _h_extent[3],
                                     const int * __restrict__ CCC,
                                     const int * __restrict__ PCL,
                                     const int * __restrict__ cell_list,
                                     %(ARGUMENTS)s){
            //cudaProfilerStart();

            //device constant copy.
            checkCudaErrors(cudaMemcpyToSymbol(_d_n, &_h_n, sizeof(_h_n)));
            checkCudaErrors(cudaMemcpyToSymbol(_d_cell_offset, &_h_cell_offset, sizeof(_h_cell_offset)));
            checkCudaErrors(cudaMemcpyToSymbol(_d_cell_array, &_h_cell_array[0], 3*sizeof(_h_cell_array[0])));
            checkCudaErrors(cudaMemcpyToSymbol(_d_extent, &_h_extent[0], 3*sizeof(_h_extent[0])));

            %(DEVICE_CONSTANT_COPY)s

            dim3 bs; bs.x = blocksize[0]; bs.y = blocksize[1]; bs.z = blocksize[2];
            dim3 ts; ts.x = threadsize[0]; ts.y = threadsize[1]; ts.z = threadsize[2];

            getLastCudaError(" %(KERNEL_NAME)s Execution failed before kernel launch. \\n");
            %(KERNEL_NAME)s_gpukernel<<<bs,ts>>>(CCC,PCL,cell_list,%(KERNEL_ARGUMENTS)s);
            checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError(" %(KERNEL_NAME)s Execution failed. \\n");

            //cudaProfilerStop();
        }
        '''

    def _generate_impl_source(self):
        """Generate the source code the actual implementation.
        """

        d = {'UNIQUENAME': self._unique_name,
             'GPU_POINTER_MAPPING': self._kernel_pointer_mapping(),
             'GPU_KERNEL': self._kernel_code,
             'ARGUMENTS': self._argnames(),
             'KERNEL_ARGUMENTS': self._kernel_argnames(),
             'KERNEL_NAME': self._kernel.name,
             'KERNEL_ARGUMENTS_DECL': self._kernel_argument_declarations(),
             'DEVICE_CONSTANT_DECELERATION': self._device_const_dec,
             'DEVICE_CONSTANT_COPY': self._device_const_copy,
             'ACCEL_VECTOR': self._get_acceleration_array(),
             'POS_VECTOR': self._get_position_array()
             }
        return self._code % d

    def _get_acceleration_array(self):
        s = '//'
        for dat_orig in self._particle_dat_dict.items():
            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
            else:
                dat = dat_orig
            if type(dat[1]) == data.ParticleDat:
                if dat[1].name == 'accelerations':
                    s = 'd_' + dat[0]
        return s

    def _get_position_array(self):
        s = '//'
        for dat_orig in self._particle_dat_dict.items():
            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
            else:
                dat = dat_orig
            if type(dat[1]) == data.ParticleDat:
                if dat[1].name == 'positions':
                    s = 'd_' + dat[0]
        return s

    def _kernel_pointer_mapping(self):
        """
        Create string for thread id and pointer mapping.
        """
        _s = ''

        space = ' ' * 14

        for dat_orig in self._particle_dat_dict.items():
            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
                _mode = dat_orig[1][1]
            else:
                dat = dat_orig
                _mode = access.RW
            argname = 'd_' + dat[0]
            loc_argname = dat[0]

            if type(dat[1]) == data.ParticleDat:
                if dat[1].name == 'positions':
                    _s += space + 'const ' + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'

                    _s += space + 'if (flag){ \n'

                    # s += space+'double r1[3];\n'
                    _s += space + 'r1[0] =' + argname + '[_iy*3]     + s[0]; \n'
                    _s += space + 'r1[1] =' + argname + '[_iy*3 + 1] + s[1]; \n'
                    _s += space + 'r1[2] =' + argname + '[_iy*3 + 2] + s[2]; \n'
                    _s += space + loc_argname + '[1] = r1;\n'

                    _s += space + '}else{ \n'
                    _s += space + loc_argname + '[1] = ' + argname + '+3*_iy;\n'
                    _s += space + '} \n'
                    _s += space + loc_argname + '[0] = _p;\n'
                elif dat[1].name == 'accelerations':
                    _s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'
                    _s += space + host.ctypes_map[dat[1].dtype] + ' dummy[3] = {0,0,0};\n'
                    _s += space + loc_argname + '[0] = _a;\n'
                    _s += space + loc_argname + '[1] = dummy;\n'


                else:
                    _s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'
                    _s += space + loc_argname + '[0] = ' + argname + '+' + str(dat[1].ncomp) + '*_ix;\n'
                    _s += space + loc_argname + '[1] = ' + argname + '+' + str(dat[1].ncomp) + '*_iy;\n'

            elif type(dat[1]) == data.ScalarArray:
                _s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ' = ' + argname + ';\n'

        return _s

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H
        #include "%(LIB_DIR)s/cuda_generic.h"

        %(INCLUDED_HEADERS)s

        extern "C"         void %(KERNEL_NAME)s_wrapper(const int blocksize[3],
                                     const int threadsize[3],
                                     const int _h_n,
                                     const int _h_cell_offset,
                                     const int _h_cell_array[3],
                                     const double _h_extent[3],
                                     const int * __restrict__ CCC,
                                     const int * __restrict__ PCL,
                                     const int * __restrict__ cell_list,
                                     %(ARGUMENTS)s);

        #endif
        '''
        d = {'UNIQUENAME': self._unique_name,
             'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir}
        return code % d

    def execute(self, dat_dict=None, static_args=None):



        """Allow alternative pointers"""
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict

        _tpb = 128
        _blocksize = (ct.c_int * 3)(int(math.ceil(self._N() / float(_tpb))), 1, 1)
        _threadsize = (ct.c_int * 3)(_tpb, 1, 1)

        _h_cell_array = (ct.c_int * 3)(self._domain.cell_array[0],
                                       self._domain.cell_array[1],
                                       self._domain.cell_array[2])

        _h_extent = (ct.c_double * 3)(self._domain.extent[0],
                                      self._domain.extent[1],
                                      self._domain.extent[2])

        args = [_blocksize,
                _threadsize,
                ct.c_int(self._N()),
                ct.c_int(cell.cell_list.cell_list[cell.cell_list.cell_list.end]),
                _h_cell_array,
                _h_extent,
                cell.cell_list.cell_contents_count.get_cuda_dat().ctypes_data,
                cell.cell_list.cell_reverse_lookup.get_cuda_dat().ctypes_data,
                cell.cell_list.cell_list.get_cuda_dat().ctypes_data
                ]

        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Add pointer arguments to launch command'''
        for dat in self._particle_dat_dict.values():
            if type(dat) is tuple:
                args.append(dat[0].get_cuda_dat().ctypes_data)
            else:
                args.append(dat.get_cuda_dat().ctypes_data)

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        method(*args)




class SimpleCudaPairLoopHalo(SimpleCudaPairLoop):



    def _code_init(self):
        self._kernel_code = self._kernel.code

        self._code = '''
        #include \"%(UNIQUENAME)s.h\"

        //device constant decelerations.
        __constant__ int _d_n;
        __constant__ int _d_cell_offset;
        __constant__ int _d_cell_array[3];

        __constant__ int cell_map[27] = %(CELL_OFFSETS)s

        %(DEVICE_CONSTANT_DECELERATION)s

        //device kernel decelerations.
        __global__ void %(KERNEL_NAME)s_gpukernel(int *CCC, int *PCL, int *cell_list, %(KERNEL_ARGUMENTS_DECL)s){

            int _ix = threadIdx.x + blockIdx.x*blockDim.x;
            if (_ix < _d_n){

                //create local store for acceleration of particle _ix.
                double _a[3];
                _a[0] = 0; _a[1] = 0; _a[2] = 0;

                //ensure positon of particle _ix is only read once.
                 const double *P[2];
                 P[0] =  %(POS_VECTOR)s + _ix*3;

                for(int cpp_i=0; cpp_i<27; cpp_i++){
                    int cpp = PCL[_ix] + cell_map[cpp_i];

                    /*
                    if (cell_list[_d_cell_offset+cpp] > -1){
                        for(int _iy = cell_list[_d_cell_offset+cpp];
                         _iy < cell_list[_d_cell_offset+cpp]+CCC[cpp];
                         _iy++){
                    */

                    int _iy = cell_list[_d_cell_offset+cpp];
                    while(_iy > -1){


                            if (_iy != _ix){
                            %(GPU_POINTER_MAPPING)s
                            %(GPU_KERNEL)s
                            }
                        _iy = cell_list[_iy];
                        }
                    //}
                }

                //Write acceleration to dat.
                %(ACCEL_VECTOR)s[_ix*3]     = _a[0];
                %(ACCEL_VECTOR)s[_ix*3 + 1] = _a[1];
                %(ACCEL_VECTOR)s[_ix*3 + 2] = _a[2];

            }
            return;
        }

        void %(KERNEL_NAME)s_wrapper(const int blocksize[3],
                                     const int threadsize[3],
                                     const int _h_n,
                                     const int _h_cell_offset,
                                     const int _h_cell_array[3],
                                     int *CCC,
                                     int *PCL,
                                     int *cell_list,
                                     %(ARGUMENTS)s){
            //cudaProfilerStart();

            //device constant copy.
            checkCudaErrors(cudaMemcpyToSymbol(_d_n, &_h_n, sizeof(_h_n)));
            checkCudaErrors(cudaMemcpyToSymbol(_d_cell_offset, &_h_cell_offset, sizeof(_h_cell_offset)));
            checkCudaErrors(cudaMemcpyToSymbol(_d_cell_array, &_h_cell_array[0], 3*sizeof(_h_cell_array[0])));

            %(DEVICE_CONSTANT_COPY)s

            dim3 bs; bs.x = blocksize[0]; bs.y = blocksize[1]; bs.z = blocksize[2];
            dim3 ts; ts.x = threadsize[0]; ts.y = threadsize[1]; ts.z = threadsize[2];

            getLastCudaError(" %(KERNEL_NAME)s Execution failed before kernel launch. \\n");
            %(KERNEL_NAME)s_gpukernel<<<bs,ts>>>(CCC,PCL,cell_list,%(KERNEL_ARGUMENTS)s);
            checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError(" %(KERNEL_NAME)s Execution failed. \\n");

            //cudaProfilerStop();
        }
        '''
    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H
        #include "%(LIB_DIR)s/cuda_generic.h"

        %(INCLUDED_HEADERS)s

        extern "C"         void %(KERNEL_NAME)s_wrapper(const int blocksize[3],
                                     const int threadsize[3],
                                     const int _h_n,
                                     const int _h_cell_offset,
                                     const int _h_cell_array[3],
                                     int * CCC,
                                     int * PCL,
                                     int *cell_list,
                                     %(ARGUMENTS)s);

        #endif
        '''
        d = {'UNIQUENAME': self._unique_name,
             'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir}
        return code % d

    def _kernel_pointer_mapping(self):
        """
        Create string for thread id and pointer mapping.
        """
        _s = ''

        space = ' ' * 14

        for dat_orig in self._particle_dat_dict.items():
            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
                _mode = dat_orig[1][1]
            else:
                dat = dat_orig
                _mode = access.RW
            argname = 'd_' + dat[0]
            loc_argname = dat[0]

            if type(dat[1]) == data.ParticleDat:
                if dat[1].name == 'positions':
                    _s += space + loc_argname + '[1] = ' + argname + '+3*_iy;\n'
                elif dat[1].name == 'accelerations':
                    _s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'
                    _s += space + host.ctypes_map[dat[1].dtype] + ' dummy[3] = {0,0,0};\n'
                    _s += space + loc_argname + '[0] = _a;\n'
                    _s += space + loc_argname + '[1] = dummy;\n'


                else:
                    _s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'
                    _s += space + loc_argname + '[0] = ' + argname + '+' + str(dat[1].ncomp) + '*_ix;\n'
                    _s += space + loc_argname + '[1] = ' + argname + '+' + str(dat[1].ncomp) + '*_iy;\n'

            elif type(dat[1]) == data.ScalarArray:
                _s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ' = ' + argname + ';\n'

        return _s

    def cell_offset_mapping(self):
        """
        Calculate cell offset mappings

        :return:
        """
        _map = ((-1,1,-1),
                (-1,-1,-1),
                (-1,0,-1),
                (0,1,-1),
                (0,-1,-1),
                (0,0,-1),
                (1,0,-1),
                (1,1,-1),
                (1,-1,-1),

                (-1,1,0),
                (-1,0,0),
                (-1,-1,0),
                (0,-1,0),
                (0,0,0),
                (0,1,0),
                (1,0,0),
                (1,1,0),
                (1,-1,0),

                (-1,0,1),
                (-1,1,1),
                (-1,-1,1),
                (0,0,1),
                (0,1,1),
                (0,-1,1),
                (1,0,1),
                (1,1,1),
                (1,-1,1))

        _s = '{'
        for ix in range(27):
            _s1 = str(_map[ix][0] + _map[ix][1] * self._domain.cell_array[0] + _map[ix][2] * self._domain.cell_array[0]* self._domain.cell_array[1])
            if ix < 26:
                _s += _s1 + ','
            else:
                _s += _s1 + '}; \n'

        return _s

    def _generate_impl_source(self):
        """Generate the source code the actual implementation.
        """

        d = {'UNIQUENAME': self._unique_name,
             'GPU_POINTER_MAPPING': self._kernel_pointer_mapping(),
             'GPU_KERNEL': self._kernel_code,
             'ARGUMENTS': self._argnames(),
             'KERNEL_ARGUMENTS': self._kernel_argnames(),
             'KERNEL_NAME': self._kernel.name,
             'KERNEL_ARGUMENTS_DECL': self._kernel_argument_declarations(),
             'DEVICE_CONSTANT_DECELERATION': self._device_const_dec,
             'DEVICE_CONSTANT_COPY': self._device_const_copy,
             'ACCEL_VECTOR': self._get_acceleration_array(),
             'POS_VECTOR': self._get_position_array(),
             'CELL_OFFSETS': self.cell_offset_mapping()
             }
        return self._code % d


    def execute(self, dat_dict=None, static_args=None):



        """Allow alternative pointers"""
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict

        _tpb = 256
        _blocksize = (ct.c_int * 3)(int(math.ceil(self._N() / float(_tpb))), 1, 1)
        _threadsize = (ct.c_int * 3)(_tpb, 1, 1)

        _h_cell_array = (ct.c_int * 3)(self._domain.cell_array[0],
                                       self._domain.cell_array[1],
                                       self._domain.cell_array[2])

        args = [_blocksize,
                _threadsize,
                ct.c_int(self._N()),
                ct.c_int(cell.cell_list.cell_list[cell.cell_list.cell_list.end]),
                _h_cell_array,
                cell.cell_list.cell_contents_count.get_cuda_dat().ctypes_data,
                cell.cell_list.cell_reverse_lookup.get_cuda_dat().ctypes_data,
                cell.cell_list.cell_list.get_cuda_dat().ctypes_data
                ]

        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Add pointer arguments to launch command'''
        for dat in self._particle_dat_dict.values():
            if type(dat) is tuple:
                args.append(dat[0].get_cuda_dat().ctypes_data)
            else:
                args.append(dat.get_cuda_dat().ctypes_data)

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        method(*args)
        
        
class SimpleCudaPairLoopHalo2D(SimpleCudaPairLoop):



    def _code_init(self):
        self._kernel_code = self._kernel.code

        self._code = '''
        #include \"%(UNIQUENAME)s.h\"

        //device constant decelerations.
        __constant__ int _d_n;
        __constant__ int _d_cell_offset;
        __constant__ int _d_cell_array[3];

        __constant__ int cell_map[27] = %(CELL_OFFSETS)s

        %(DEVICE_CONSTANT_DECELERATION)s


        //device kernel decelerations.
        __global__ void %(KERNEL_NAME)s_gpukernel(const int * __restrict__ CCC, const int * __restrict__ PCL, const int * __restrict__ cell_list, %(KERNEL_ARGUMENTS_DECL)s){

            const int _ix = threadIdx.x + blockIdx.x*blockDim.x;
            double u[1]; u[0] = 0.0;

            if (_ix < _d_n){
                //create local store for acceleration of particle _ix.

                double _a[3] = {0};
                //_a[0] = 0; _a[1] = 0; _a[2] = 0;
                double *A[2]; A[0] = _a;

                const double *P[2];
                P[0] =  %(POS_VECTOR)s + _ix*3;

                for(int cpp_i=0; cpp_i<27; cpp_i++){
                    int cpp = PCL[_ix] + cell_map[cpp_i];

                        for(unsigned int _iy = cell_list[_d_cell_offset+cpp];
                         _iy < cell_list[_d_cell_offset+cpp]+CCC[cpp];
                         _iy++){

                            if (_iy != _ix){
                            %(GPU_POINTER_MAPPING)s

                            %(GPU_KERNEL)s



                            }
                         }

                }

                //Write acceleration to dat.

                %(ACCEL_VECTOR)s[_ix*3]     = _a[0];
                %(ACCEL_VECTOR)s[_ix*3 + 1] = _a[1];
                %(ACCEL_VECTOR)s[_ix*3 + 2] = _a[2];

            }

            u[0] = warpReduceSumDouble(u[0]);

            if (  (int)(threadIdx.x & (warpSize - 1)) == 0)
            {
                atomicAddDouble(&d_u[0], u[0]);
            }




            return;
        }

        void %(KERNEL_NAME)s_wrapper(const int blocksize[3],
                                     const int threadsize[3],
                                     const int _h_n,
                                     const int _h_cell_offset,
                                     const int _h_cell_array[3],
                                     const int * __restrict__ CCC,
                                     const int * __restrict__ PCL,
                                     const int * __restrict__ cell_list,
                                     %(ARGUMENTS)s){
            //cudaProfilerStart();

            //device constant copy.
            checkCudaErrors(cudaMemcpyToSymbol(_d_n, &_h_n, sizeof(_h_n)));
            checkCudaErrors(cudaMemcpyToSymbol(_d_cell_offset, &_h_cell_offset, sizeof(_h_cell_offset)));
            checkCudaErrors(cudaMemcpyToSymbol(_d_cell_array, &_h_cell_array[0], 3*sizeof(_h_cell_array[0])));

            %(DEVICE_CONSTANT_COPY)s

            dim3 bs; bs.x = blocksize[0]; bs.y = blocksize[1]; bs.z = blocksize[2];
            dim3 ts; ts.x = threadsize[0]; ts.y = threadsize[1]; ts.z = threadsize[2];

            getLastCudaError(" %(KERNEL_NAME)s Execution failed before kernel launch. \\n");
            %(KERNEL_NAME)s_gpukernel<<<bs,ts>>>(CCC,PCL,cell_list,%(KERNEL_ARGUMENTS)s);
            checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError(" %(KERNEL_NAME)s Execution failed. \\n");

            //cudaProfilerStop();
        }
        '''

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''
        #ifndef %(UNIQUENAME)s_H
        #define %(UNIQUENAME)s_H %(UNIQUENAME)s_H
        #include "%(LIB_DIR)s/cuda_generic.h"

        %(INCLUDED_HEADERS)s

        extern "C"         void %(KERNEL_NAME)s_wrapper(const int blocksize[3],
                                     const int threadsize[3],
                                     const int _h_n,
                                     const int _h_cell_offset,
                                     const int _h_cell_array[3],
                                     const int * __restrict__ CCC,
                                     const int * __restrict__ PCL,
                                     const int * __restrict__ cell_list,
                                     %(ARGUMENTS)s);

        #endif
        '''
        d = {'UNIQUENAME': self._unique_name,
             'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': runtime.LIB_DIR.dir}
        return code % d


    def _kernel_pointer_mapping(self):
        """
        Create string for thread id and pointer mapping.
        """
        _s = ''

        space = ' ' * 14

        for dat_orig in self._particle_dat_dict.items():
            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
                _mode = dat_orig[1][1]
            else:
                dat = dat_orig
                _mode = access.RW
            argname = 'd_' + dat[0]
            loc_argname = dat[0]

            if type(dat[1]) == data.ParticleDat:
                if dat[1].name == 'positions':
                    _s += space + loc_argname + '[1] = ' + argname + '+3*_iy;\n'
                    # _s += 'memcpy(&_p2[0], &' + argname + '[_iy*3], sizeof(double)*3); \n'
                elif dat[1].name == 'forces':
                    _s += space + host.ctypes_map[dat[1].dtype] + ' dummy[3] = {0,0,0};\n'
                    _s += space + loc_argname + '[1] = dummy;\n'


                else:
                    _s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + '[2];\n'
                    _s += space + loc_argname + '[0] = ' + argname + '+' + str(dat[1].ncomp) + '*_ix;\n'
                    _s += space + loc_argname + '[1] = ' + argname + '+' + str(dat[1].ncomp) + '*_iy;\n'

            elif type(dat[1]) == data.ScalarArray:
                pass
                # _s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ' = ' + argname + ';\n'

        return _s

    def cell_offset_mapping(self):
        """
        Calculate cell offset mappings

        :return:
        """
        _map = ((-1,1,-1),
                (-1,-1,-1),
                (-1,0,-1),
                (0,1,-1),
                (0,-1,-1),
                (0,0,-1),
                (1,0,-1),
                (1,1,-1),
                (1,-1,-1),

                (-1,1,0),
                (-1,0,0),
                (-1,-1,0),
                (0,-1,0),
                (0,0,0),
                (0,1,0),
                (1,0,0),
                (1,1,0),
                (1,-1,0),

                (-1,0,1),
                (-1,1,1),
                (-1,-1,1),
                (0,0,1),
                (0,1,1),
                (0,-1,1),
                (1,0,1),
                (1,1,1),
                (1,-1,1))

        _s = '{'
        for ix in range(27):
            _s1 = str(_map[ix][0] + _map[ix][1] * self._domain.cell_array[0] + _map[ix][2] * self._domain.cell_array[0]* self._domain.cell_array[1])
            if ix < 26:
                _s += _s1 + ','
            else:
                _s += _s1 + '}; \n'

        return _s

    def _generate_impl_source(self):
        """Generate the source code the actual implementation.
        """

        d = {'UNIQUENAME': self._unique_name,
             'GPU_POINTER_MAPPING': self._kernel_pointer_mapping(),
             'GPU_KERNEL': self._kernel_code,
             'ARGUMENTS': self._argnames(),
             'KERNEL_ARGUMENTS': self._kernel_argnames(),
             'KERNEL_NAME': self._kernel.name,
             'KERNEL_ARGUMENTS_DECL': self._kernel_argument_declarations(),
             'DEVICE_CONSTANT_DECELERATION': self._device_const_dec,
             'DEVICE_CONSTANT_COPY': self._device_const_copy,
             'ACCEL_VECTOR': self._get_acceleration_array(),
             'POS_VECTOR': self._get_position_array(),
             'CELL_OFFSETS': self.cell_offset_mapping()
             }
        return self._code % d


    def execute(self, dat_dict=None, static_args=None):

        """Allow alternative pointers"""
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict

        _tpb = 128
        _blocksize = (ct.c_int * 3)(int(math.ceil(self._N() / float(_tpb))), 1, 1)
        _threadsize = (ct.c_int * 3)(_tpb, 1, 1)

        _h_cell_array = (ct.c_int * 3)(self._domain.cell_array[0],
                                       self._domain.cell_array[1],
                                       self._domain.cell_array[2])

        args = [_blocksize,
                _threadsize,
                ct.c_int(self._N()),
                ct.c_int(cell.cell_list.cell_list[cell.cell_list.cell_list.end]),
                _h_cell_array,
                cell.cell_list.cell_contents_count.get_cuda_dat().ctypes_data,
                cell.cell_list.cell_reverse_lookup.get_cuda_dat().ctypes_data,
                cell.cell_list.cell_list.get_cuda_dat().ctypes_data
                ]

        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Add pointer arguments to launch command'''
        for dat in self._particle_dat_dict.values():
            if type(dat) is tuple:
                args.append(dat[0].get_cuda_dat().ctypes_data)
            else:
                args.append(dat.get_cuda_dat().ctypes_data)

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        method(*args)