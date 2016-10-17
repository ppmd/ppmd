"""
cuda looping code
"""

# system level
import ctypes
import math
import cgen

# package level
import ppmd.access as access
import ppmd.host as host
import ppmd.runtime as runtime
import ppmd.opt as opt

# CUDA level
import cuda_build
import cuda_runtime
import cuda_generation
import cuda_base
import cuda_data


class _ParticleLoopOLD(object):
    def __init__(self, kernel, dat_dict, n=None ,types_map=None):
        self._types_map = types_map
        self._kernel = kernel
        self._dat_dict = dat_dict

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
        for arg in self._dat_dict.items():
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
        self._gcode['HCODE_lib_dir'].add(cuda_runtime.LIB_DIR)
        self._gcode['KCODE_gpu_kernel'].add(self._kernel.code)

    def _generate_impl_source(self):
        """Generate the source code the actual implementation.
        """

        return self._code % self._gcode

    def execute(self, n=None, dat_dict=None, static_args=None, threads=1024):

        """Allow alternative pointers"""
        if dat_dict is not None:
            self._dat_dict = dat_dict

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
        for dat in self._dat_dict.values():
            if type(dat) is tuple:
                args.append(dat[0].ctypes_data)
            else:
                args.append(dat.ctypes_data)

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']

        method(*args)

        '''afterwards access descriptors'''
        for dat_orig in self._dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])
            else:
                dat_orig.ctypes_data_post()




    def _generate_dynamics(self):

        host_args = ''
        host_k_call_args = ''

        for i, dat_orig in enumerate(self._dat_dict.items()):

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


# #############################################################################
# --------------------------------- AST ---------------------------------------
# #############################################################################








def generate_reduction_final_stage(symbol_external, symbol_internal, dat):
    """
    Reduce arrays here

    :arg string symbol_external: variable name for shared library
    :arg string symbol_internal: variable name for kernel.
    :arg cuda_data.data: :class:`~cuda_data.ParticleDat` or :class:`~cuda_data.ScalarArray` cuda_data.object in question.

    :return: string for initialisation code.
    """
    _nl = '\n'
    _space = ' ' * 14

    _s = _nl

    _s += _space + 'for(int _iz = 0; _iz < ' + str(dat.ncomp) + '; _iz++ ){ \n'
    _s += 2 * _space + symbol_internal + '[_iz] = warpReduceSumDouble(' + symbol_internal + '[_iz]); \n'
    _s += _space + '}\n'



    _s += _space + '__shared__ ' + host.ctypes_map[dat.dtype] + ' _d_red_' + symbol_internal + '[' + str(dat.ncomp) + ']; \n'
    _s += _space + 'if (  (int)(threadIdx.x & (warpSize - 1)) == 0){ \n'
    _s += 2 * _space + 'for(int _iz = 0; _iz < ' + str(dat.ncomp) + '; _iz++ ){ \n'
    _s += 3 * _space +'_d_red_' + symbol_internal + '[_iz] = 0; \n'
    _s += _space + '}} __syncthreads(); \n'



    # reduce into the shared dat.
    _s += _space + 'if (  (int)(threadIdx.x & (warpSize - 1)) == 0){ \n'
    _s += 2 * _space + 'for(int _iz = 0; _iz < ' + str(dat.ncomp) + '; _iz++ ){ \n'
    _s += 3 * _space +'atomicAddDouble(&_d_red_' + symbol_internal + '[_iz], ' + symbol_internal + '[_iz]); \n'
    _s += _space + '}} __syncthreads(); \n'

    _s += _space + 'if (threadIdx.x == 0){ \n'
    _s += 2 * _space + 'for(int _iz = 0; _iz < ' + str(dat.ncomp) + '; _iz++ ){ \n'
    _s += 3 * _space +'atomicAddDouble(&' + symbol_external + '[_iz], _d_red_' + symbol_internal + '[_iz]); \n'
    _s += _space + '}} \n'

    return _s + '\n'



def Restrict(keyword, symbol):
    return str(keyword) + ' ' + str(symbol)


class ParticleLoop(object):

    def __init__(self, kernel, dat_dict, n=None, types_map=None):

        self._types_map = types_map
        self._kernel = kernel
        self._dat_dict = dat_dict

        # set compiler as NVCC default
        self._cc = cuda_build.NVCC

        self.loop_timer = opt.LoopTimer()

        self._components = {'LIB_PAIR_INDEX_0': '_i',
                            'LIB_NAME': str(self._kernel.name) + '_wrapper'}

        self._group = None

        for pd in self._dat_dict.items():

            if issubclass(type(pd[1][0]), cuda_data.ParticleDat):
                if pd[1][0].group is not None:
                    self._group = pd[1][0].group
                    break

        #start code creation
        self._generate()

        # Create library
        self._lib = cuda_build.simple_lib_creator(
            self._generate_header_source(),
            self._components['LIB_SRC'],
            self._components['LIB_NAME']
        )


    def _generate(self):


        self._generate_lib_specific_args()
        self._generate_per_dat()
        self._generate_map_macros()
        self._generate_kernel_headers()

        self._generate_kernel_func()

        self._generate_lib_func()
        self._generate_lib_src()

        #print 60*"-"
        #print self._components['LIB_SRC']
        #print 60*"-"



    def _generate_lib_specific_args(self):
        self._components['LIB_ARG_DECLS'] = [
            cgen.Const(
                cgen.Pointer(
                    cgen.Value(host.int32_str,
                               Restrict(
                                   self._cc.restrict_keyword,'_H_BLOCKSIZE'
                               )
                               )
                )
            ),
            cgen.Const(
                cgen.Pointer(
                    cgen.Value(host.int32_str,
                               Restrict(
                                   self._cc.restrict_keyword,'_H_THREADSIZE'
                               )
                               )
                )
            ),
            cgen.Const(cgen.Value(host.int32_str, '_H_N_LOCAL')),
            self.loop_timer.get_cpp_arguments_ast()
        ]


    # -------------------------------------------------------------------------


    def _generate_per_dat(self):


        # =================== DICT INIT ===============================

        self._components['KERNEL_ARG_DECLS'] = [
            cgen.Const(cgen.Value(host.int32_str, '_D_N_LOCAL'))
        ]

        self._components['KERNEL_LIB_ARG_DECLS'] = []

        self._components['KERNEL_STRUCT_TYPEDEFS'] = cgen.Module([
            cgen.Comment('#### Structs generated per ParticleDat ####')
        ])

        self._components['LIB_KERNEL_CALL'] = cgen.Module([cgen.Comment('#### Kernel call ####')])
        kernel_call_symbols = ['_H_N_LOCAL']


        self._components['KERNEL_SCATTER'] = cgen.Module([cgen.Comment('#### kernel scatter ####')])
        self._components['KERNEL_GATHER'] = cgen.Module([cgen.Comment('#### kernel gather ####')])
        self._components['IF_SCATTER'] = cgen.Module([cgen.Comment('#### if scatter ####')])
        self._components['IF_GATHER'] = cgen.Module([cgen.Comment('#### if gather ####')])
        self._components['KERNEL_MAPPING'] = cgen.Module([cgen.Comment('#### kernel symbol mapping ####')])



        # =================== Static Args ===============================
        if self._kernel.static_args is not None:

            for i, datt in enumerate(self._kernel.static_args.items()):

                ksym = datt[0]
                ktype = datt[1]

                # Add to kernel args
                g = cgen.Const(cgen.Value(host.ctypes_map[ktype], ksym))
                self._components['KERNEL_ARG_DECLS'].append(g)
                self._components['KERNEL_LIB_ARG_DECLS'].append(g)
                kernel_call_symbols.append(ksym)

        # =================== Dynamic Args ===============================
        for i, datt in enumerate(self._dat_dict.items()):
            assert type(datt[1]) is tuple, "Access descriptors not found"

            dati = datt[1][0]
            ksym = datt[0]
            dsym = 'd_' + ksym
            kacc = datt[1][1]

            # add to lib args
            kernel_lib_arg = cgen.Pointer(cgen.Value(host.ctypes_map[dati.dtype],
                                          Restrict(self._cc.restrict_keyword, ksym))
                                      )


            if issubclass(type(dati), cuda_base.Array):


                # KERNEL ARGS DECLS -----------------------------
                kernel_arg = cgen.Pointer(cgen.Value(host.ctypes_map[dati.dtype],
                                              Restrict(self._cc.restrict_keyword, dsym))
                                          )
                if not kacc.write:
                    kernel_arg = cgen.Const(kernel_arg)
                self._components['KERNEL_ARG_DECLS'].append(kernel_arg)

                # KERNEL CALL SYMS -----------------------------
                kernel_call_symbols.append(ksym)


                # KERNEL GATHER/SCATTER START ------------------
                if not kacc.incremented:
                    a = cgen.Initializer(
                        cgen.Pointer(cgen.Value(
                            host.ctypes_map[dati.dtype], ksym
                        )),
                        dsym
                    )
                    if not kacc.write:
                        a = cgen.Const(a)
                    self._components['IF_GATHER'].append(a)
                else:
                    if kacc is access.INC0:
                        s = '{0}'
                    else:
                        s = '{'
                        for cx in range(dati.ncomp - 1):
                            s += dsym + '[' + str(cx) + '], '
                        s += dsym + '[' + str(dati.ncomp - 1) + ']}'

                    a = cgen.Initializer(
                        cgen.Value(
                            host.ctypes_map[dati.dtype],
                            ksym + '[' + str(dati.ncomp) +']'
                        ),
                        s
                    )

                    self._components['IF_GATHER'].append(a)

                    # add the scatter code
                    self._components['IF_SCATTER'].append(
                        cgen.Line(
                            generate_reduction_final_stage(
                                dsym,
                                ksym,
                                dati)
                        )
                    )
                    # KERNEL GATHER/SCATTER END ------------------



            elif issubclass(type(dati), cuda_base.Matrix):

                # KERNEL ARGS DECLS, STRUCT DECLS ----------------

                dtype = dati.dtype
                ti = cgen.Pointer(cgen.Value(cgen.dtype_to_ctype(dtype),
                                             Restrict(self._cc.restrict_keyword,'i')))
                if not kacc.write:
                    ti = cgen.Const(ti)
                typename = '_'+ksym+'_t'
                self._components['KERNEL_STRUCT_TYPEDEFS'].append(
                    cgen.Typedef(cgen.Struct('', [ti], typename))
                )


                # add to kernel args
                kernel_arg = cgen.Pointer(cgen.Value(host.ctypes_map[dati.dtype],
                                              Restrict(self._cc.restrict_keyword, dsym))
                                          )
                if not kacc.write:
                    kernel_arg = cgen.Const(kernel_arg)
                self._components['KERNEL_ARG_DECLS'].append(kernel_arg)


                # KERNEL CALL SYMS -----------------------------
                kernel_call_symbols.append(ksym)



                # KERNEL GATHER/SCATTER START ------------------
                nc = str(dati.ncomp)
                _ishift = '+' + self._components['LIB_PAIR_INDEX_0'] + '*' + nc

                isym = dsym + _ishift
                g = cgen.Value(typename, ksym)
                g = cgen.Initializer(g, '{ ' + isym + '}')

                self._components['KERNEL_MAPPING'].append(g)
                # KERNEL GATHER/SCATTER END ------------------


            # END OF IF ------------------------

            # add to lib args
            if not kacc.write:
                kernel_lib_arg = cgen.Const(kernel_lib_arg)
            self._components['KERNEL_LIB_ARG_DECLS'].append(kernel_lib_arg)



        # KERNEL CALL SYMS -----------------------------

        kernel_call_symbols_s = ''
        for sx in kernel_call_symbols:
            kernel_call_symbols_s += sx +','
        kernel_call_symbols_s=kernel_call_symbols_s[:-1]


        self._components['LIB_KERNEL_CALL'].append(cgen.Module([
            cgen.Value('dim3', '_B'),
            cgen.Value('dim3', '_T'),
            cgen.Assign('_B.x', '_H_BLOCKSIZE[0]'),
            cgen.Assign('_B.y', '_H_BLOCKSIZE[1]'),
            cgen.Assign('_B.z', '_H_BLOCKSIZE[2]'),
            cgen.Assign('_T.x', '_H_THREADSIZE[0]'),
            cgen.Assign('_T.y', '_H_THREADSIZE[1]'),
            cgen.Assign('_T.z', '_H_THREADSIZE[2]')
        ]))


        self._components['LIB_KERNEL_CALL'].append(cgen.Line(
            'k_'+self._kernel.name+'<<<_B,_T>>>(' + kernel_call_symbols_s + ');'
        ))
        self._components['LIB_KERNEL_CALL'].append(cgen.Line(
            'checkCudaErrors(cudaDeviceSynchronize());'
        ))


    # -------------------------------------------------------------------------


    def _generate_map_macros(self):
        g = cgen.Module([cgen.Comment('#### KERNEL_MAP_MACROS ####')])
        for i, dat in enumerate(self._dat_dict.items()):
            if issubclass(type(dat[1][0]), cuda_base.Array):
                g.append(cgen.Define(dat[0]+'(x)', '('+dat[0]+'[(x)])'))
            if issubclass(type(dat[1][0]), cuda_base.Matrix):
                g.append(cgen.Define(dat[0]+'(y)', dat[0]+'.i[(y)]'))

        self._components['KERNEL_MAP_MACROS'] = g


    def _generate_kernel_func(self):


        if_block = cgen.If(
            self._components['LIB_PAIR_INDEX_0']+'<_D_N_LOCAL',
            cgen.Block([
                self._components['KERNEL_GATHER'],
                self._components['KERNEL_MAPPING'],
                cgen.Line(self._kernel.code),
                self._components['KERNEL_SCATTER']
                ])
            )


        func = cgen.Block([
            cgen.Initializer(
                    cgen.Const(
                    cgen.Value(
                        host.int32_str,
                        self._components['LIB_PAIR_INDEX_0']
                    )),
                    'threadIdx.x + blockIdx.x*blockDim.x'
            ),
            self._components['IF_GATHER'],
            if_block,
            self._components['IF_SCATTER']
        ])


        self._components['KERNEL_FUNC'] = cgen.FunctionBody(

            cgen.FunctionDeclaration(
                cgen.DeclSpecifier(
                    cgen.Value("void", 'k_' + self._kernel.name), '__global__'
                ),
                self._components['KERNEL_ARG_DECLS']
            ),
                func
            )


    def _generate_kernel_headers(self):

        s = [
            cgen.Include(cuda_runtime.LIB_DIR + '/cuda_generic.h',
                         system=False)
        ]

        if self._kernel.headers is not None:
            for x in self._kernel.headers:
                s.append(x.ast)

        s.append(self.loop_timer.get_cpp_headers_ast())
        self._components['KERNEL_HEADERS'] = cgen.Module(s)



    def _generate_lib_func(self):
        block = cgen.Block([
            self.loop_timer.get_cpp_pre_loop_code_ast(),
            self._components['LIB_KERNEL_CALL'],
            self.loop_timer.get_cpp_post_loop_code_ast()
        ])


        self._components['LIB_FUNC'] = cgen.FunctionBody(
            cgen.FunctionDeclaration(
                cgen.Value("void", self._components['LIB_NAME'])
            ,
                self._components['LIB_ARG_DECLS'] + self._components['KERNEL_LIB_ARG_DECLS']
            ),
                block
            )



    def _generate_lib_src(self):


        self._components['LIB_SRC'] = cgen.Module([
            self._components['KERNEL_STRUCT_TYPEDEFS'],
            self._components['KERNEL_MAP_MACROS'],
            cgen.Comment('#### Kernel function ####'),
            self._components['KERNEL_FUNC'],
            cgen.Comment('#### Library function ####'),
            self._components['LIB_FUNC']
        ])



    def _generate_header_source(self):
        """Generate the source code of the header file.
        Returns the source code for the header file.
        """
        code = '''
        %(INCLUDED_HEADERS)s

        #include "%(LIB_DIR)s/generic.h"

        extern "C" %(FUNC_DEC)s
        '''
        d = {'INCLUDED_HEADERS': str(self._components['KERNEL_HEADERS']),
             'FUNC_DEC': str(self._components['LIB_FUNC'].fdecl),
             'LIB_DIR': runtime.LIB_DIR}
        return code % d


    def execute(self, n=None, dat_dict=None, static_args=None, threads=1024):

        """Allow alternative pointers"""
        if dat_dict is not None:
            self._dat_dict = dat_dict


        if n is None:
            if self._group is not None:
                n = self._group.npart_local

        assert n is not None, "No n found"

        if n <= threads:

            _blocksize = (ctypes.c_int * 3)(1, 1, 1)
            _threadsize = (ctypes.c_int * 3)(threads, 1, 1)

        else:
            _blocksize = (ctypes.c_int * 3)(int(math.ceil(n / float(threads))), 1, 1)
            _threadsize = (ctypes.c_int * 3)(threads, 1, 1)


        args = [ctypes.byref(_blocksize),
                ctypes.byref(_threadsize),
                ctypes.c_int(n),
                self.loop_timer.get_python_parameters()
                ]

        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Pass access descriptor to dat'''
        for dat_orig in self._dat_dict.values():
            dat_orig[0].ctypes_data_access(dat_orig[1], pair=False)


        '''Add pointer arguments to launch command'''
        for dat in self._dat_dict.values():
            args.append(dat[0].ctypes_data)


        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._components['LIB_NAME']]

        method(*args)

        '''afterwards access descriptors'''
        for dat_orig in self._dat_dict.values():
            dat_orig[0].ctypes_data_post(dat_orig[1])


















