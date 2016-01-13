"""
cuda looping code
"""

# system level
import ctypes
import math

# package level
import ppmd.access as access
import ppmd.host as host

# CUDA level
import cuda_build
import cuda_data
import cuda_runtime


class ParticleLoop(object):
    def __init__(self, types_map, kernel, particle_dat_dict):
        self._types_map = types_map
        self._kernel = kernel
        self._particle_dat_dict = particle_dat_dict

        # set compiler as NVCC default
        self._cc = cuda_build.NVCC

        #start code creation
        self._static_arg_init()
        self._code_init()

        self._lib = cuda_build.simple_lib_creator(self._generate_header_source(), self._generate_impl_source(), 'ParticleLoop')


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

    @staticmethod
    def _mode_arg_dec_str(mode):
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

            if type(dat[1]) == cuda_data.ParticleDat:
                s += self._mode_arg_dec_str(_mode) + host.ctypes_map[dat[1].dtype] + ' * ' + self._cc.restrict_keyword + ' d_' + loc_argname + ','
            if type(dat[1]) == cuda_data.ScalarArray:
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

            if type(dat[1]) == cuda_data.ParticleDat:
                _s += space + host.ctypes_map[dat[1].dtype] + ' *' + loc_argname + ';\n'
                _s += space + loc_argname + ' = ' + argname + '+' + str(dat[1].ncomp) + '*_ix;\n'

        return _s

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''
        #include "%(LIB_DIR)s/cuda_generic.h"

        %(INCLUDED_HEADERS)s

        extern "C" void %(KERNEL_NAME)s_wrapper(const int blocksize[3], const int threadsize[3], const int _h_n, %(ARGUMENTS)s);

        '''

        d = {'INCLUDED_HEADERS': self._included_headers(),
             'KERNEL_NAME': self._kernel.name,
             'ARGUMENTS': self._argnames(),
             'LIB_DIR': cuda_runtime.LIB_DIR.dir}

        return code % d


    def _code_init(self):

        self._code = '''

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

        self._gcode = {'KCODE_pre_if': cuda_build.Code(),
                       'KCODE_pre_loop': cuda_build.Code(),
                       'KCODE_gpu_pointer': cuda_build.Code(),
                       'KCODE_post_loop': cuda_build.Code(),
                       'KCODE_post_if': cuda_build.Code(),

                       'HCODE_device_constant_dec': cuda_build.Code(),
                       'HCODE_kernel_name': cuda_build.Code(),
                       'HCODE_arguments': cuda_build.Code(),
                       'HCODE_device_constant_copy': cuda_build.Code(),
                       'HCODE_kernel_arguments': cuda_build.Code()}


    def _generate_impl_source(self):
        """Generate the source code the actual implementation.
        """

        d = {'GPU_POINTER_MAPPING': self._kernel_pointer_mapping(),
             'GPU_KERNEL': self._kernel.code,
             'ARGUMENTS': self._argnames(),
             'KERNEL_ARGUMENTS': self._kernel_argnames(),
             'KERNEL_NAME': self._kernel.name,
             'KERNEL_ARGUMENTS_DECL': self._kernel_argument_declarations(),
             'DEVICE_CONSTANT_DECELERATION': self._device_const_dec,
             'DEVICE_CONSTANT_COPY': self._device_const_copy}

        return self._code % d

    def execute(self, dat_dict=None, static_args=None):

        """Allow alternative pointers"""
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict

        if self._particle_dat_dict.values()[0].npart < 1025:
            _blocksize = (ctypes.c_int * 3)(1, 1, 1)
            _threadsize = (ctypes.c_int * 3)(self._particle_dat_dict.values()[0].npart, 1, 1)

        else:
            _blocksize = (ctypes.c_int * 3)(int(math.ceil(self._particle_dat_dict.values()[0].npart / 1024.)), 1, 1)
            _threadsize = (ctypes.c_int * 3)(1024, 1, 1)


        args = [_blocksize, _threadsize, ctypes.c_int(self._particle_dat_dict.values()[0].npart)]

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






