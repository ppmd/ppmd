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
import cuda_runtime
import cuda_generation


class ParticleLoop(object):
    def __init__(self, kernel, particle_dat_dict, types_map=None):
        self._types_map = types_map
        self._kernel = kernel
        self._particle_dat_dict = particle_dat_dict

        # set compiler as NVCC default
        self._cc = cuda_build.NVCC

        #start code creation
        self._generate()

        # Create library
        self._lib = cuda_build.simple_lib_creator(self._generate_header_source(), self._generate_impl_source(), kernel.name + '_ParticleLoop')

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

        if self._kernel.static_args is not None:

            for ix in self._kernel.static_args.items():
                self._gcode['HCODE_device_constant_dec'].add_line('__constant__ ' + host.ctypes_map[ix[1]] + ' ' + ix[0] + ';')
                self._gcode['HCODE_device_constant_copy'].add_line('checkCudaErrors(cudaMemcpyToSymbol(' + ix[0] + ', &h_' + ix[0] + ', sizeof(h_' + ix[0] + ')));')
                self._gcode['HCODE_arguments'] += 'const ' + host.ctypes_map[ix[1]] + ' h_' + ix[0] + ','

    def _kernel_argnames(self):
        """
        Comma separated string of argument names to be passed to kernel launch.
        """
        args = ''
        for arg in self._particle_dat_dict.items():
            args += 'd_' + arg[0] + ','
        return args[:-1]

    def _generate_header_source(self):
        """Generate the source code of the header file.

        Returns the source code for the header file.
        """
        code = '''
        #include "%(HCODE_lib_dir)s/cuda_generic.h"

        %(HCODE_included_headers)s

        extern "C" void %(HCODE_kernel_name)s_wrapper(const int blocksize[3], const int threadsize[3], const int _h_n, %(HCODE_arguments)s);

        '''

        return code % self._gcode

    def _code_init(self):

        # Create the template
        self._code = '''

        //device constant decelerations.
        __constant__ int _d_n;
        %(HCODE_device_constant_dec)s

        //device kernel decelerations.
        __global__ void %(HCODE_kernel_name)s_gpukernel(%(KCODE_arguments)s){

            int _ix = threadIdx.x + blockIdx.x*blockDim.x;

            %(KCODE_pre_if)s

            if (_ix < _d_n){

                %(KCODE_gpu_pointer)s

                %(KCODE_gpu_kernel)s

            }

            %(KCODE_post_if)s

        }

        void %(HCODE_kernel_name)s_wrapper(const int blocksize[3], const int threadsize[3], const int _h_n, %(HCODE_arguments)s) {

            //device constant copy.
            checkCudaErrors(cudaMemcpyToSymbol(_d_n, &_h_n, sizeof(_h_n)));
            %(HCODE_device_constant_copy)s

            dim3 bs; bs.x = blocksize[0]; bs.y = blocksize[1]; bs.z = blocksize[2];
            dim3 ts; ts.x = threadsize[0]; ts.y = threadsize[1]; ts.z = threadsize[2];


            %(HCODE_kernel_name)s_gpukernel<<<bs,ts>>>(%(HCODE_kernel_arguments)s);
            checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError(" %(HCODE_kernel_name)s Execution failed. \\n");

            return;
        }
        '''

        # Initialise the generated code part

        self._gcode = {'KCODE_arguments': cuda_build.Code(),
                       'KCODE_pre_if': cuda_build.Code(),
                       'KCODE_pre_loop': cuda_build.Code(),
                       'KCODE_gpu_pointer': cuda_build.Code(),
                       'KCODE_gpu_kernel': cuda_build.Code(),
                       'KCODE_post_loop': cuda_build.Code(),
                       'KCODE_post_if': cuda_build.Code(),

                       'HCODE_lib_dir': cuda_build.Code(),
                       'HCODE_included_headers': cuda_build.Code(),
                       'HCODE_device_constant_dec': cuda_build.Code(),
                       'HCODE_kernel_name': cuda_build.Code(),
                       'HCODE_arguments': cuda_build.Code(),
                       'HCODE_device_constant_copy': cuda_build.Code(),
                       'HCODE_kernel_arguments': cuda_build.Code()}

    def _generate(self):
        """
        Generate code.
        """

        self._code_init()
        self._base_generation()
        self._static_arg_init()
        self._generate_dynamics()

    def _base_generation(self):
        """
        Create the basics in the code generation.
        :return:
        """

        self._gcode['HCODE_included_headers'].add(self._included_headers())
        self._gcode['HCODE_kernel_name'].add(self._kernel.name)
        self._gcode['HCODE_lib_dir'].add(cuda_runtime.LIB_DIR.dir)
        self._gcode['KCODE_gpu_kernel'].add(self._kernel.code)

    def _generate_impl_source(self):
        """Generate the source code the actual implementation.
        """

        return self._code % self._gcode

    def execute(self, n=None, dat_dict=None, static_args=None, threads=1024):

        """Allow alternative pointers"""
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict

        assert n is not None, "No number of particles passed"


        if n <= threads:

            _blocksize = (ctypes.c_int * 3)(1, 1, 1)
            _threadsize = (ctypes.c_int * 3)(threads, 1, 1)

        else:
            _blocksize = (ctypes.c_int * 3)(int(math.ceil(n / float(threads))), 1, 1)
            _threadsize = (ctypes.c_int * 3)(threads, 1, 1)


        args = [_blocksize, _threadsize, ctypes.c_int(n)]

        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Add pointer arguments to launch command'''
        for dat in self._particle_dat_dict.values():
            if type(dat) is tuple:
                args.append(dat[0].ctypes_data)
            else:
                args.append(dat.ctypes_data)

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        method(*args)

    def _generate_dynamics(self):

        host_args = ''
        host_k_call_args = ''

        for i, dat_orig in enumerate(self._particle_dat_dict.items()):

            if type(dat_orig[1]) is tuple:
                dat = dat_orig[0], dat_orig[1][0]
                _mode = dat_orig[1][1]
            else:
                dat = dat_orig
                _mode = access.RW

            self._gcode['KCODE_pre_if'] += cuda_generation.create_local_reduction_vars_arrays(symbol_external='d_' + dat[0],
                                                                                              symbol_internal=dat[0],
                                                                                              dat=dat[1],
                                                                                              access_type=_mode)


            self._gcode['KCODE_gpu_pointer'] += cuda_generation.generate_map(pair=False,
                                                                             symbol_external='d_' + dat[0],
                                                                             symbol_internal=dat[0],
                                                                             dat=dat[1],
                                                                             access_type=_mode,
                                                                             n3_disable_dats=[])

            self._gcode['KCODE_post_if'] += cuda_generation.generate_reduction_final_stage(symbol_external='d_' + dat[0],
                                                                                           symbol_internal=dat[0],
                                                                                           dat=dat[1],
                                                                                           access_type=_mode)

            host_args += cuda_generation.create_host_function_argument_decleration(symbol='d_' + dat[0],
                                                                                   dat=dat[1],
                                                                                   mode=_mode,
                                                                                   cc=self._cc
                                                                                   ) + ','

            host_k_call_args += 'd_' + dat[0] + ','

        self._gcode['HCODE_arguments'] += host_args[:-1]
        self._gcode['KCODE_arguments'] += host_args[:-1]
        self._gcode['HCODE_kernel_arguments'] += host_k_call_args[:-1]
