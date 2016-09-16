"""
Contains the code to generate cuda pairloops
"""

# system level
import ctypes
import math

# package level
import ppmd.access as access
import ppmd.host as host
import ppmd.logic as logic


# CUDA level
import cuda_build
import cuda_runtime
import cuda_generation
import cuda_data
import cuda_cell



class PairLoopNeighbourList(object):

    _neighbour_list_dict = {}

    def __init__(self, kernel=None, dat_dict=None, shell_cutoff=None):
        assert kernel is not None, "no kernel passed"
        assert dat_dict is not None, "No particle dat dict passed"

        self._kernel = kernel
        self._particle_dat_dict = dat_dict





        if type(shell_cutoff) is not logic.Distance:
            shell_cutoff = logic.Distance(shell_cutoff)
        self.shell_cutoff = shell_cutoff

        self._group = None

        for pd in self._particle_dat_dict.items():
            if issubclass(type(pd[1][0]), cuda_data.PositionDat):
                self._group = pd[1][0].group
                break

        assert self._group is not None, "No cell to particle map found"
        new_decomp_flag = self._group.get_domain().cell_decompose(self.shell_cutoff.value)

        if new_decomp_flag:
            self._group.get_cell_to_particle_map().create()

        self._key = (self.shell_cutoff, self._group.get_domain(), self._group.get_position_dat())

        _nd = PairLoopNeighbourList._neighbour_list_dict
        if not self._key in _nd.keys() or new_decomp_flag:
            _nd[self._key] = cuda_cell.NeighbourListLayerBased(
                occ_matrix=self._group.get_cell_to_particle_map(),
                cutoff=self.shell_cutoff.value
            )


        self._nl = _nd[self._key]


        self._gcode = None

        self._cc = cuda_build.NVCC

        self._generate()
        # Create library
        self._lib = cuda_build.simple_lib_creator(self._generate_header_source(), self._generate_impl_source(), 'PairLoopNeighbourList')[self._kernel.name + '_wrapper']

    def _generate_header_source(self):
        return '''
        #include "%(HCODE_lib_dir)s/cuda_generic.h"

        %(HCODE_included_headers)s

        extern "C" void %(HCODE_kernel_name)s_wrapper(const int blocksize[3],
                                           const int threadsize[3],
                                           const int _h_n,
                                           const int _h_max_neigh,
                                           const cuda_Matrix<int> d_nm,
                                           %(HCODE_arguments)s);

        ''' % self._gcode

    def _generate_impl_source(self):
        return self._code % self._gcode

    def _code_init(self):
        self._code = '''

        //device const decelerations
        __constant__ int _d_n;
        __constant__ int _d_max_neigh;
        %(HCODE_device_constant_dec)s

        //device kernel decelerations
        __global__ void %(HCODE_kernel_name)s_gpukernel(const int* __restrict__ d_nm, %(KCODE_arguments)s){

            const int _ix = threadIdx.x + blockIdx.x*blockDim.x;

            %(KCODE_pre_if)s


            if (_ix < _d_n){


                %(KCODE_pre_loop)s

                //printf("PL ix=%%d, N_neigh=%%d \\n", _ix, d_nm[_ix]);


                for(int _idy = 1; _idy <= d_nm[_ix]; _idy++){


                    const int _iy = d_nm[_ix + _d_n*_idy];

                    //printf("PL ix=%%d, iy=%%d \\n", _ix, _iy);

                    %(KCODE_gpu_pointer)s


                    %(KCODE_gpu_kernel)s

                }

                %(KCODE_post_loop)s
            }

            %(KCODE_post_if)s

            return;
        }


        void %(HCODE_kernel_name)s_wrapper(const int blocksize[3],
                                           const int threadsize[3],
                                           const int _h_n,
                                           const int _h_max_neigh,
                                           const cuda_Matrix<int> d_nm,
                                           %(HCODE_arguments)s){

            //device constant copy.
            checkCudaErrors(cudaMemcpyToSymbol(_d_n, &_h_n, sizeof(_h_n)));
            checkCudaErrors(cudaMemcpyToSymbol(_d_max_neigh, &_h_max_neigh, sizeof(_h_n)));
            %(HCODE_device_constant_copy)s

            dim3 bs; bs.x = blocksize[0]; bs.y = blocksize[1]; bs.z = blocksize[2];
            dim3 ts; ts.x = threadsize[0]; ts.y = threadsize[1]; ts.z = threadsize[2];


            %(HCODE_kernel_name)s_gpukernel<<<bs,ts>>>(d_nm.ptr, %(HCODE_kernel_arguments)s);
            checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError(" %(HCODE_kernel_name)s Execution failed. \\n");

            return;
        }

        '''

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

    def _included_headers(self):
        """Return names of included header files."""
        s = ''
        if self._kernel.headers is not None:
            s += '\n'
            for x in self._kernel.headers:
                s += '#include \"' + x + '\" \n'
        return s

    def _generate(self):
        self._code_init()
        self._base_generation()
        self._static_arg_init()
        self._generate_dynamics()


    def _base_generation(self):
        self._gcode['HCODE_included_headers'].add(self._included_headers())
        self._gcode['HCODE_kernel_name'].add(self._kernel.name)
        self._gcode['HCODE_lib_dir'].add(cuda_runtime.LIB_DIR)
        self._gcode['KCODE_gpu_kernel'].add(self._kernel.code)

    def _static_arg_init(self):
        if self._kernel.static_args is not None:
            for ix in self._kernel.static_args.items():
                self._gcode['HCODE_device_constant_dec'].add_line('__constant__ ' + host.ctypes_map[ix[1]] + ' ' + ix[0] + ';')
                self._gcode['HCODE_device_constant_copy'].add_line('checkCudaErrors(cudaMemcpyToSymbol(' + ix[0] + ', &h_' + ix[0] + ', sizeof(h_' + ix[0] + ')));')
                self._gcode['HCODE_arguments'] += 'const ' + host.ctypes_map[ix[1]] + ' h_' + ix[0] + ','

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

            '''
            if dat[1].name == 'forces':
                _dd = [dat[1]]
            else:
                _dd = []
            '''

            _dd = []

            self._gcode['KCODE_pre_if'] +=\
                cuda_generation.create_local_reduction_vars_arrays(
                    symbol_external='d_' + dat[0],
                    symbol_internal=dat[0],
                    dat=dat[1],
                    access_type=_mode
                )

            self._gcode['KCODE_pre_loop'] +=\
                cuda_generation.create_pre_loop_map_matrices(
                    pair=True,
                    symbol_external='d_' + dat[0],
                    symbol_internal=dat[0],
                    dat=dat[1],
                    access_type=_mode
                )

            self._gcode['KCODE_gpu_pointer'] +=\
                cuda_generation.generate_map(
                    pair=True,
                    symbol_external='d_' + dat[0],
                    symbol_internal=dat[0],
                    dat=dat[1],
                    access_type=_mode,
                    n3_disable_dats=_dd
                )

            self._gcode['KCODE_post_loop'] += \
                cuda_generation.create_post_loop_map_matrices(
                    pair=True,
                    symbol_external='d_' + dat[0],
                    symbol_internal=dat[0],
                    dat=dat[1],
                    access_type=_mode
                )

            self._gcode['KCODE_post_if'] += \
                cuda_generation.generate_reduction_final_stage(
                    symbol_external='d_' + dat[0],
                    symbol_internal=dat[0],
                    dat=dat[1],
                    access_type=_mode
                )

            host_args += \
                cuda_generation.create_host_function_argument_decleration(
                    symbol='d_' + dat[0],
                    dat=dat[1],
                    mode=_mode,
                    cc=self._cc
                ) + ','

            host_k_call_args += 'd_' + dat[0] + ','

        self._gcode['HCODE_arguments'] += host_args[:-1]
        self._gcode['KCODE_arguments'] += host_args[:-1]
        self._gcode['HCODE_kernel_arguments'] += host_k_call_args[:-1]


    def execute(self, n=None, dat_dict=None, static_args=None, threads=256):


        cell2part = self._group.get_cell_to_particle_map()
        cell2part.check()


        """Allow alternative pointers"""
        if dat_dict is not None:
            self._particle_dat_dict = dat_dict

        if n is None:
            n = self._group.npart_local

        if n <= threads:

            _blocksize = (ctypes.c_int * 3)(1, 1, 1)
            _threadsize = (ctypes.c_int * 3)(threads, 1, 1)

        else:
            _blocksize = (ctypes.c_int * 3)(int(math.ceil(n / float(threads))), 1, 1)
            _threadsize = (ctypes.c_int * 3)(threads, 1, 1)



        dargs = []
        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                dargs.append(dat)


        '''Pass access descriptor to dat'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_access(dat_orig[1])

        '''Add pointer arguments to launch command'''
        for dat in self._particle_dat_dict.values():
            if type(dat) is tuple:
                dargs.append(dat[0].ctypes_data)
            else:
                dargs.append(dat.ctypes_data)


        if cell2part.version_id > self._nl.version_id:
            #print "CUDA rebuild NL"
            self._nl.update()


        args2 = [_blocksize, _threadsize, ctypes.c_int(n), ctypes.c_int(self._nl.max_neigbours_per_particle), self._nl.list.struct]


        args = args2 + dargs

        '''Execute the kernel over all particle pairs.'''
        self._lib(*args)



        '''afterwards access descriptors'''
        for dat_orig in self._particle_dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])
            else:
                dat_orig.ctypes_data_post()














