# system level
from __future__ import division, print_function
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import cgen
import ctypes

# package level

from ppmd import data, runtime, host, access, modules, opt
from ppmd.lib import build
from ppmd.pairloop.neighbourlist import Restrict, scatter_matrix

from ppmd.modules.dsl_seq_comp import DSLSeqComp
from ppmd.modules.dsl_stride_comp import DSLStrideComp
from ppmd.modules.dsl_struct_comp import DSLStructComp

from ppmd.modules.dsl_cell_gather_scatter import *
from ppmd.modules.dsl_record_local import *
from ppmd.modules.dsl_cell_list_loop import DSLCellListIter

from ppmd.modules.dsl_kernel_symbols import *

_offsets = (
    (-1,1,-1),
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
    (1,-1,1)
)




INT64 = ctypes.c_int64

class SubCellByCellOMP(object):

    def __init__(self, kernel=None, dat_dict=None, shell_cutoff=None):

        self._dat_dict = access.DatArgStore(
            self._get_allowed_types(),
            dat_dict
        )

        self._cc = build.TMPCC
        self._kernel = kernel
        self.shell_cutoff = shell_cutoff
        
        self.loop_timer = modules.code_timer.LoopTimer()
        self.wrapper_timer = opt.Timer(runtime.TIMER)
        self.list_timer = opt.Timer(runtime.TIMER)

        self._gather_space = host.ThreadSpace(100, ctypes.c_uint8)
        self._generate()

        self._offset_list = host.Array(ncomp=27, dtype=ctypes.c_int)

        self._lib = build.simple_lib_creator(self._generate_header_source(),
                                             self._components['LIB_SRC'],
                                             self._kernel.name,
                                             CC=self._cc)
        self._group = None

        for pd in self._dat_dict.items():
            if issubclass(type(pd[1][0]), data.PositionDat):
                self._group = pd[1][0].group
                break

        #assert self._group is not None, "No cell to particle map found"
        if self._group is not None:
            self._make_cell_list(self._group)

        self._kernel_execution_count = INT64(0)
        self._invocations = 0

        self._jstore = [host.Array(ncomp=100, dtype=ctypes.c_int) for tx in \
                        range(runtime.NUM_THREADS)]
        

    def _make_cell_list(self, group):
        # if flag is true then a new cell list was created
        flag = group.cell_decompose(self.shell_cutoff)
        self._make_offset_list(group)

    def _make_offset_list(self, group):
        ca = group.domain.cell_array
        for ofi, ofs in enumerate(_offsets):
            self._offset_list[ofi] = ofs[0] + ca[0]*ofs[1] + ca[0]*ca[1]*ofs[2]

    @staticmethod
    def _get_allowed_types():
        return {
            data.ScalarArray: (access.READ,),
            data.ParticleDat: access.all_access_types,
            data.PositionDat: access.all_access_types,
            data.GlobalArrayClassic: (access.INC_ZERO, access.INC, access.READ),
            data.GlobalArrayShared: (access.INC_ZERO, access.INC, access.READ),
        }

    def _generate(self):
        self._init_components()
        self._generate_kernel()
        self._generate_kernel_gather()
        self._generate_particle_dat_c()
        self._generate_lib_specific_args()
        self._generate_kernel_arg_decls()
        self._generate_kernel_func()
        self._generate_kernel_headers()

        self._generate_kernel_call()
        self._generate_kernel_scatter()

        self._generate_lib_inner_loop_block()
        self._generate_lib_inner_loop()

        self._generate_lib_outer_loop()
        self._generate_lib_func()
        self._generate_lib_src()
    
    def _generate_kernel(self):
        k = DSLKernelSymSub(kernel=self._kernel.code)

        self._components['KERNEL'] = k

    def _generate_kernel_func(self):
        self._components['KERNEL_FUNC'] = cgen.FunctionBody(
            cgen.FunctionDeclaration(
                cgen.DeclSpecifier(
                    cgen.Value("void", 'k_' + self._kernel.name), 'inline'
                ),
                self._components['KERNEL_ARG_DECLS']
            ),
                cgen.Block([
                    cgen.Line(self._components['KERNEL'].kernel)
                ])
            )

    def _generate_kernel_headers(self):
        s = self._components['LIB_HEADERS']
        if self._kernel.headers is not None:
            if hasattr(self._kernel.headers, "__iter__"):
                for x in self._kernel.headers:
                    s.append(x.ast)
            else:
                s.append(self._kernel.headers.ast)

        s.append(self.loop_timer.get_cpp_headers_ast())
        self._components['KERNEL_HEADERS'] = cgen.Module(s)

    def _generate_lib_inner_loop_block(self):
        # generate j gather
        #'J_GATHER'
        cj = self._components['LIB_CELL_INDEX_1']



        j_gather = cgen.Module([
            cgen.Comment('#### Pre kernel j gather ####'),
        ])
        
        
        inner_l = []
        src_sym = '_tmp_jgpx'
        dst_sym = self._components['CCC_1']

        # add dats to omp shared and init global array reduction
        for i, dat in enumerate(self._dat_dict.items()):

            obj = dat[1][0]
            mode = dat[1][1]
            symbol = dat[0]

            if issubclass(type(obj), data.ParticleDat):
                tsym = self._components['PARTICLE_DAT_PARTITION'].jdict[symbol]
                inner_l.append(DSLStrideGather(
                    symbol, tsym, obj.ncomp, src_sym, dst_sym,
                    self._components['CCC_MAX']
                ))


        inner_l.append(cgen.Line(dst_sym+'++;'))        
        
        
        inner = cgen.Module(inner_l)
        g = self._components['CELL_LIST_ITER'](src_sym, cj, inner)
        
        j_gather.append(cgen.Initializer(cgen.Value('INT64', dst_sym),'0'))
        j_gather.append(g)

        self._components['J_GATHER'] = j_gather



    def _generate_kernel_scatter(self):
        kernel_scatter = cgen.Module([cgen.Comment('#### Post kernel scatter ####')])

        ci = self._components['LIB_CELL_INDEX_0']

        inner_l = []
        src_sym = '_sgpx'
        dst_sym = '_shpx'
        # add dats to omp shared and init global array reduction
        for i, dat in enumerate(self._dat_dict.items()):

            obj = dat[1][0]
            mode = dat[1][1]
            symbol = dat[0]

            if issubclass(type(obj), data.ParticleDat) and mode.write:
                tsym = self._components['PARTICLE_DAT_PARTITION'].idict[symbol]
                inner_l.append(DSLStrideScatter(
                    tsym, symbol, obj.ncomp, dst_sym, src_sym,
                    self._components['CCC_MAX']
                ))
        

        inner_l.append(cgen.Line(dst_sym+'++;')) 
        inner = cgen.Module(inner_l)
        g = self._components['CELL_LIST_ITER'](src_sym, ci, inner)
        
        kernel_scatter.append(cgen.Initializer(cgen.Value('INT64', dst_sym),'0'))
        kernel_scatter.append(g)


        self._components['LIB_KERNEL_SCATTER'] = kernel_scatter

    def _init_components(self):
        self._components = {
            'PARTICLE_DAT_C': dict(),
            'PARTICLE_DAT_PARTITION': None,
            'LIB_PAIR_INDEX_0': '_i',
            'LIB_PAIR_INDEX_1': '_j',
            'LIB_CELL_INDEX_0': '_ci',
            'LIB_CELL_INDEX_1': '_cj',
            'CCC_0': '_gpx',
            'CCC_1': '_jgpx',
            'LIB_CELL_CX': '_CX',
            'LIB_CELL_CY': '_CY',
            'LIB_CELL_CZ': '_CZ',
            'N_CELL_X': '_N_CELL_X',
            'N_CELL_Y': '_N_CELL_Y',
            'N_CELL_Z': '_N_CELL_Z',
            'N_CELL_PAD': '_N_CELL_PAD',
            'N_LOCAL' : '_N_LOCAL',
            'I_LOCAL_SYM': '_I_LOCAL_COUNT',
            'J_GATHER' : None,
            'LIB_NAME': str(self._kernel.name) + '_wrapper',
            'LIB_HEADERS': [cgen.Include('omp.h', system=True),],
            'OMP_THREAD_INDEX_SYM': '_threadid',
            'OMP_SHARED_SYMS': ['_CELL_LIST', '_OFFSET', '_CRL', '_CCC',
                                '_JSTORE', '_GATHER_SPACE'],
            'CELL_LIST_ITER': None,
            'TMP_INDEX': '_TMP_INDEX',
            'CCC_MAX': '_MAX_CELL',
            'EXEC_COUNT': '_EXEC_COUNT',
            'KERNEL_GATHER': '',
            'KERNEL_SCATTER': ''
        }

        self._components['CELL_LIST_ITER'] = DSLCellListIter(
            '_CELL_LIST', '_LIST_OFFSET'
        )

    def _generate_lib_specific_args(self):
        cp = self._components
        ncx =  cp['N_CELL_X']
        ncy =  cp['N_CELL_Y']
        ncz =  cp['N_CELL_Z']
        npad = cp['N_CELL_PAD']
        nloc = cp['N_LOCAL']
        exec_count = cp['EXEC_COUNT']

        self._components['LIB_ARG_DECLS'] = [
            cgen.Const(cgen.Value(host.int32_str, '_NUM_THREADS')),
            cgen.Const(cgen.Value(host.int64_str, ncx)),
            cgen.Const(cgen.Value(host.int64_str, ncy)),
            cgen.Const(cgen.Value(host.int64_str, ncz)),
            cgen.Const(cgen.Value(host.int64_str, npad)),
            cgen.Const(cgen.Value(host.int32_str, nloc)),
            cgen.Const(cgen.Value(host.int32_str, '_LIST_OFFSET')),
            cgen.Const(
                cgen.Pointer(
                    cgen.Value(host.int32_str,
                               Restrict(
                                   self._cc.restrict_keyword,'_CELL_LIST'
                               )
                    )
                )
            ),
            cgen.Const(
                cgen.Pointer(
                    cgen.Value(host.int32_str,
                               Restrict(self._cc.restrict_keyword, '_CRL')),
                )
            ),
            cgen.Const(
                cgen.Pointer(
                    cgen.Value(host.int32_str,
                               Restrict(self._cc.restrict_keyword, '_CCC')),
                )
            ),
            cgen.Const(
                cgen.Pointer(
                    cgen.Value(host.int32_str,
                               Restrict(self._cc.restrict_keyword, '_OFFSET')),
                )
            ),
            cgen.Pointer(
                cgen.Pointer(
                    cgen.Value(host.int32_str,
                               Restrict(self._cc.restrict_keyword, '_JSTORE')),
                )
            ),
            cgen.Pointer(
                cgen.Pointer(
                    cgen.Value(host.uint8_str,
                               Restrict(self._cc.restrict_keyword,
                                   '_GATHER_SPACE')),
                )
            ),
            cgen.Const(cgen.Value(host.int64_str,
                self._components['CCC_MAX'])),
            cgen.Pointer(cgen.Value(host.int64_str, exec_count)),
            self.loop_timer.get_cpp_arguments_ast()
        ]
    def _generate_lib_func(self):
        block = cgen.Block([
            self.loop_timer.get_cpp_pre_loop_code_ast(),
            self._components['LIB_OUTER_LOOP'],
            self.loop_timer.get_cpp_post_loop_code_ast()
        ])
        self._components['LIB_FUNC'] = cgen.FunctionBody(
            cgen.FunctionDeclaration(
                cgen.Value("void", self._components['LIB_NAME'])
            ,
                self._components['LIB_ARG_DECLS'] + \
                    self._components['KERNEL_LIB_ARG_DECLS']
            ),
                block
            )

    def _generate_lib_src(self):
        self._components['LIB_SRC'] = cgen.Module([
            self._components['KERNEL_STRUCT_TYPEDEFS'],
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
        #include <cstdint>
        #include "%(LIB_DIR)s/generic.h"
        #define INT64 int64_t
        extern "C" %(FUNC_DEC)s
        '''
        d = {'INCLUDED_HEADERS': str(self._components['KERNEL_HEADERS']),
             'FUNC_DEC': str(self._components['LIB_FUNC'].fdecl),
             'LIB_DIR': runtime.LIB_DIR}
        return code % d

    def _update_opt(self):
        opt.PROFILE[
            self.__class__.__name__+':'+self._kernel.name+':list_timer'
        ] = self.list_timer.time()

        opt.PROFILE[
            self.__class__.__name__+':'+self._kernel.name+':execute_internal'
        ] = self.loop_timer.time

        opt.PROFILE[
            self.__class__.__name__+':'+self._kernel.name+\
                ':kernel_execution_count'
        ] =  self._kernel_execution_count.value
    
    def _generate_particle_dat_c(self):
        c = self._components
        for i, dat in enumerate(self._dat_dict.items()):
            obj = dat[1][0]
            mode = dat[1][1]
            symbol = dat[0]
            if issubclass(type(obj), host.Matrix):

                dsc = DSLStrideComp(
                            sym=symbol,
                            i_gather_sym=self._components[
                                'PARTICLE_DAT_PARTITION'].idict[symbol],
                            j_gather_sym=self._components[
                                'PARTICLE_DAT_PARTITION'].jdict[symbol],
                            ctype=host.ctypes_map[obj.dtype],
                            const=True if not mode.write else False,
                            ncomp=obj.ncomp,
                            i_index=self._components['LIB_PAIR_INDEX_0'],
                            j_index=self._components['LIB_PAIR_INDEX_1'],
                            stride=self._components['CCC_MAX']
                )

                c['KERNEL'].sub_sym(symbol+'.i', dsc.isymbol)
                c['KERNEL'].sub_sym(symbol+'.j', dsc.jsymbol)

                c['PARTICLE_DAT_C'][symbol] = dsc


    def _generate_kernel_arg_decls(self):

        _kernel_arg_decls = []
        _kernel_lib_arg_decls = []
        _kernel_structs = cgen.Module([
            cgen.Comment('#### Structs generated per ParticleDat ####')
        ])

        if self._kernel.static_args is not None:
            for i, dat in enumerate(self._kernel.static_args.items()):
                arg = cgen.Const(cgen.Value(host.ctypes_map[dat[1]], dat[0]))
                _kernel_arg_decls.append(arg)
                _kernel_lib_arg_decls.append(arg)

        for i, dat in enumerate(self._dat_dict.items()):

            assert type(dat[1]) is tuple, "Access descriptors not found"
            obj = dat[1][0]
            mode = dat[1][1]
            symbol = dat[0]

            kernel_lib_arg = cgen.Pointer(
                cgen.Value(
                    host.ctypes_map[obj.dtype],
                    Restrict(self._cc.restrict_keyword, symbol)
            ))

            if issubclass(type(obj), data.GlobalArrayClassic):
                kernel_lib_arg = cgen.Pointer(kernel_lib_arg)

            if issubclass(type(obj), host._Array):
                kernel_arg = cgen.Pointer(
                    cgen.Value(
                        host.ctypes_map[obj.dtype],
                        Restrict(self._cc.restrict_keyword, symbol)
                ))

                if not mode.write:
                    kernel_arg = cgen.Const(kernel_arg)
                _kernel_arg_decls.append(kernel_arg)

                if mode.write is True:
                    assert issubclass(type(obj), data.GlobalArrayClassic), \
                        "global array must be a thread safe type for \
                        write access. Type is:" + str(type(obj))


            elif issubclass(type(dat[1][0]), host.Matrix):
                gen = self._components['PARTICLE_DAT_C'][symbol]
                _kernel_structs.append(gen.header)
                _kernel_arg_decls.append(gen.kernel_arg_decl[0])
                _kernel_arg_decls.append(gen.kernel_arg_decl[1])

            if not dat[1][1].write:
                kernel_lib_arg = cgen.Const(kernel_lib_arg)

            _kernel_lib_arg_decls.append(kernel_lib_arg)

        self._components['KERNEL_ARG_DECLS'] = _kernel_arg_decls
        self._components['KERNEL_LIB_ARG_DECLS'] = _kernel_lib_arg_decls
        self._components['KERNEL_STRUCT_TYPEDEFS'] = _kernel_structs


    def _get_static_lib_args(self, static_args):
        args = []

        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not " \
                                            "passed to loop."
            for dat in self._kernel.static_args.get_args(static_args):
                args.append(dat)

        return args

    def _init_dat_lib_args(self, dats):
        """
        The halo exchange process may reallocate all in the group. Hence this
        function loops over all dats to ensure pointers collected in the second
        pass over the dats are valid.
        """
        for dat_orig in self._dat_dict.values(dats):
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_access(dat_orig[1], pair=True)

    def _generate_kernel_gather(self):
        cp = self._components
        cx = cp['LIB_CELL_CX']
        cy = cp['LIB_CELL_CY']
        cz = cp['LIB_CELL_CZ']

        ncx =cp['N_CELL_X']
        ncy =cp['N_CELL_Y']
        ncz =cp['N_CELL_Z']
        ci = cp['LIB_CELL_INDEX_0']

        kernel_gather = cgen.Module([
            cgen.Comment('#### Pre kernel gather ####'),
            # compute the linear cell index
            cgen.Initializer(cgen.Const(cgen.Value('INT64', ci)),
                cx + '+' + ncx +'*('+cy+'+'+ncy+'*'+cz+')'),
            # get the thread index
            cgen.Initializer(cgen.Const(cgen.Value(
                'int', self._components['OMP_THREAD_INDEX_SYM'])),
                'omp_get_thread_num()')
        ])
        
        # partition this threads space for temporary vars
        self._components['PARTICLE_DAT_PARTITION'] = \
            DSLPartitionTempSpace(self._dat_dict,
                    self._components['CCC_MAX'],
                    '_GATHER_SPACE[_threadid]',
                    extras=((cp['TMP_INDEX'], 1, INT64),))
        
        kernel_gather.append(self._components['PARTICLE_DAT_PARTITION'].ptr_init)
        
        src_sym = '__tmp_gpx'
        dst_sym = cp['CCC_0']

        record_local = DSLRecordLocal(
            ind_sym=src_sym,
            nlocal_sym=cp['N_LOCAL'],
            store_sym=cp['PARTICLE_DAT_PARTITION'].idict[cp['TMP_INDEX']],
            store_ind_sym=dst_sym,
            count_sym=cp['I_LOCAL_SYM']
        )
        kernel_gather.append(record_local[0])
        inner_l = [record_local[1]]

        # add dats to omp shared and init global array reduction
        shared_syms = self._components['OMP_SHARED_SYMS']
        for i, dat in enumerate(self._dat_dict.items()):
 
            obj = dat[1][0]
            mode = dat[1][1]
            symbol = dat[0]
            shared_syms.append(symbol)

            if issubclass(type(obj), data.GlobalArrayClassic):
                isym = symbol+'_c'
                val = symbol+'['+ self._components['OMP_THREAD_INDEX_SYM'] +']'

                g = cgen.Pointer(cgen.Value(host.ctypes_map[obj.dtype], isym))
                if not mode.write:
                    g = cgen.Const(g)
                g = cgen.Initializer(g, val)

                kernel_gather.append(g)

            if issubclass(type(obj), data.ParticleDat):
                tsym = cp['PARTICLE_DAT_PARTITION'].idict[symbol]
                inner_l.append(DSLStrideGather(
                    symbol, tsym, obj.ncomp, src_sym, dst_sym,
                    self._components['CCC_MAX']
                ))
        

        inner_l.append(cgen.Line(dst_sym+'++;'))        
        
        
        inner = cgen.Module(inner_l)
        g = self._components['CELL_LIST_ITER'](src_sym, ci, inner)
        
        kernel_gather.append(cgen.Initializer(cgen.Value('INT64', dst_sym),'0'))
        kernel_gather.append(g)
        
        # skip cell if there are not local particles
        kernel_gather.append(
            cgen.If(
                cp['I_LOCAL_SYM']+'==0',
                cgen.Block((cgen.Line('continue;'),))
            )
        )


        self._components['LIB_KERNEL_GATHER'] = kernel_gather
        


    def _generate_kernel_call(self):

        kernel_call = cgen.Module([cgen.Comment(
            '#### Kernel call arguments ####')])
        kernel_call_symbols = []
        if self._kernel.static_args is not None:
            for i, dat in enumerate(self._kernel.static_args.items()):
                kernel_call_symbols.append(dat[0])
        
        for i, dat in enumerate(self._dat_dict.items()):

            obj = dat[1][0]
            mode = dat[1][1]
            symbol = dat[0]

            if issubclass(type(obj), data.GlobalArrayClassic):
                kernel_call_symbols.append(symbol+'_c')
            elif issubclass(type(obj), host._Array):
                kernel_call_symbols.append(symbol)
            elif issubclass(type(obj), host.Matrix):
                g = self._components['PARTICLE_DAT_C'][symbol]
                kernel_call_symbols.append(g.kernel_arg)
                kernel_call.append(g.kernel_create_j_arg)
                self._components['KERNEL_GATHER'] += g.kernel_create_i_arg
                self._components['KERNEL_SCATTER'] += g.kernel_create_i_scatter

            else:
                print("ERROR: Type not known")
        

        kernel_call.append(cgen.Comment('#### Kernel call ####'))

        kernel_call_symbols_s = ''
        for sx in kernel_call_symbols:
            kernel_call_symbols_s += sx +','
        kernel_call_symbols_s=kernel_call_symbols_s[:-1]

        kernel_call.append(cgen.Line(
            'k_'+self._kernel.name+'(' + kernel_call_symbols_s + ');'
        ))

        self._components['LIB_KERNEL_CALL'] = kernel_call


    def _generate_lib_inner_loop(self):
        
        c = self._components
        i = c['LIB_PAIR_INDEX_0']
        j = c['LIB_PAIR_INDEX_1']
        ccc_i = c['CCC_0']
        ccc_j = c['CCC_1']
        ci = c['LIB_CELL_INDEX_0']
        cj = c['LIB_CELL_INDEX_1']
        nloc = c['N_LOCAL']
        ec = '_'+c['EXEC_COUNT']

        iif = c['PARTICLE_DAT_PARTITION'].idict[c['TMP_INDEX']]
        def ifnothalo(b):
            return cgen.Block((cgen.If(iif+'['+i+']<'+nloc, b),))

        kg = self._components['KERNEL_GATHER']
        ks = self._components['KERNEL_SCATTER']


        loop_other = cgen.Block((cgen.For(
            'INT64 ' + i + '=0', i+'<'+ccc_i, i+'++',
            ifnothalo(cgen.Block(
                (
                    cgen.Line(kg),
                    cgen.For(
                        'INT64 ' + j + '=0', j+'<'+ccc_j, j+'++',
                        cgen.Block(self._components['LIB_KERNEL_CALL'])
                    ),
                    cgen.Line(ks),
                    cgen.Line(ec+'+='+ccc_j+';')
                )
            ))
        ),))


        loop_same = cgen.Block((cgen.For(
            'INT64 ' + i + '=0', i+'<'+ccc_i, i+'++',
            ifnothalo(
                cgen.Block((
                cgen.Line(kg),
                cgen.For(
                    'INT64 ' + j + '=0', j+'<'+i, j+'++',
                    cgen.Block(self._components['LIB_KERNEL_CALL'])
                ),
                cgen.For(
                    'INT64 ' + j + '=1+' + i, j+'<'+ccc_j, j+'++',
                    cgen.Block(self._components['LIB_KERNEL_CALL'])
                ),
                cgen.Line(ks),
                cgen.Line(ec+'+='+ccc_j+'-1;')
                ))
            )
        ),))

        cell_cond = cgen.If(
            ci+'=='+cj,
            loop_same,
            loop_other
        )


        b = cgen.Block((
            cgen.Line('const INT64 {jcell} = {icell} + _OFFSET[_k];'.format(
                jcell=self._components['LIB_CELL_INDEX_1'],
                icell=self._components['LIB_CELL_INDEX_0']
            )),
            self._components['J_GATHER'],
            cell_cond
        ))

        self._components['LIB_INNER_LOOP'] = cgen.Module([
                cgen.For('int _k=0', '_k<27', '_k++', b),
            ])

    def _generate_lib_outer_loop(self):

        block = cgen.Block([self._components['LIB_KERNEL_GATHER'],
                            self._components['LIB_INNER_LOOP'],
                            self._components['LIB_KERNEL_SCATTER']])

        cx = self._components['LIB_CELL_CX']
        cy = self._components['LIB_CELL_CY']
        cz = self._components['LIB_CELL_CZ']

        ncx = self._components['N_CELL_X']
        ncy = self._components['N_CELL_Y']
        ncz = self._components['N_CELL_Z']
        
        exec_count = self._components['EXEC_COUNT']
        red_exec_count = '_' + exec_count

        npad = self._components['N_CELL_PAD']
        

        shared = ''
        for sx in self._components['OMP_SHARED_SYMS']:
            shared+= sx+','
        shared = shared[:-1]
        pragma = cgen.Pragma('omp parallel for default(none) reduction(+:' + \
            red_exec_count +') schedule(dynamic) collapse(3) ' + \
            'shared(' + shared + ')')
        if runtime.OMP_NUM_THREADS is None:
            pragma = cgen.Comment(pragma)

        loop = cgen.Module([
            cgen.Line('omp_set_num_threads(_NUM_THREADS);'),
            cgen.Line('INT64 ' + red_exec_count + ' = 0;'),
            pragma,
            # cellx loop
            cgen.For('INT64 ' + cx + '=' + npad,
                cx + '<' + ncx + '-' + npad,
                cx+'++',
                cgen.Block(
                    [
                        cgen.For('INT64 ' + cy + '=' + npad,
                            cy + '<' + ncy + '-' + npad,
                            cy+'++',
                            cgen.Block(
                                (
                                    cgen.For('INT64 ' + cz + '=' + npad,
                                        cz + '<' + ncz + '-' + npad,
                                        cz+'++',
                                        block
                                    ),
                                )
                            )
                        ),
                    ]
                )
            ),
            cgen.Line('*'+exec_count+' += ' + red_exec_count + ';')
        ])

        self._components['LIB_OUTER_LOOP'] = loop


    def _init_jstore(self, cell2part):
        n = cell2part.max_cell_contents_count * 27
        if self._jstore[0].ncomp < n:
            self._jstore = [host.Array(ncomp=100+n, dtype=ctypes.c_int) for tx\
                            in range(runtime.NUM_THREADS)]

        return (ctypes.POINTER(ctypes.c_int) * runtime.NUM_THREADS)(*[
            self._jstore[tx].ctypes_data for tx in range(runtime.NUM_THREADS)
        ])

    def _get_class_lib_args(self, cell2part):
        assert ctypes.c_int == cell2part.cell_list.dtype
        assert ctypes.c_int == cell2part.cell_reverse_lookup.dtype
        assert ctypes.c_int == cell2part.cell_contents_count.dtype
        jstore = self._init_jstore(cell2part)
        offset = cell2part.cell_list.end - cell2part.domain.cell_count

        return [
            ctypes.c_int(runtime.NUM_THREADS),
            INT64(cell2part.cell_array[0]),
            INT64(cell2part.cell_array[1]),
            INT64(cell2part.cell_array[2]),
            INT64(1),
            ctypes.c_int(cell2part.num_particles),
            ctypes.c_int(offset),
            cell2part.cell_list.ctypes_data,
            cell2part.cell_reverse_lookup.ctypes_data,
            cell2part.cell_contents_count.ctypes_data,
            self._offset_list.ctypes_data,
            ctypes.byref(jstore),
            self._gather_space.ctypes_data,
            INT64(cell2part.max_cell_contents_count),
            ctypes.byref(self._kernel_execution_count),
            self.loop_timer.get_python_parameters()
        ]

    def _get_dat_lib_args(self, dats):
        args = []
        for dat_orig in self._dat_dict.values(dats):
            assert type(dat_orig) is tuple
            obj = dat_orig[0]
            mode = dat_orig[1]
            if issubclass(type(obj), data.GlobalArrayClassic):
                args.append(
                    obj.ctypes_data_access(mode, pair=True, threaded=True)
                )
            else:
                args.append(obj.ctypes_data_access(mode, pair=True))
        return args

    def _post_execute_dats(self, dats):
        for dat_orig in self._dat_dict.values(dats):
            assert type(dat_orig) is tuple
            obj = dat_orig[0]
            mode = dat_orig[1]
            if issubclass(type(obj), data.GlobalArrayClassic):
                obj.ctypes_data_post(mode, threaded=True)
            else:
                obj.ctypes_data_post(mode)
    
    def _prepare_tmp_space(self, max_size):
        
        req_bytes = self._components['PARTICLE_DAT_PARTITION'].req_bytes * \
                max_size

        if self._gather_space.n < req_bytes:
            self._gather_space = host.ThreadSpace(n=req_bytes+100,
                    dtype=ctypes.c_uint8)


    def execute(self, n=None, dat_dict=None, static_args=None):

        _group = self._group # could be None
        if _group is None:
            for pd in self._dat_dict.items(dat_dict):
                if issubclass(type(pd[1][0]), data.PositionDat):
                    _group = pd[1][0].group
                    break
            self._make_cell_list(_group)

        assert _group is not None, "no group"
        cell2part = _group.get_cell_to_particle_map()
        cell2part.check()
        self._make_cell_list(_group)

        args = []
        # Add static arguments to launch command
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not " \
                                            "passed to loop."
            args += self._kernel.static_args.get_args(static_args)

        # Add pointer arguments to launch command
        self._init_dat_lib_args(dat_dict)
        args+=self._get_dat_lib_args(dat_dict)
        
        # get max cell contents after halo exchange
        self._prepare_tmp_space(cell2part.max_cell_contents_count)
        

        # Rebuild neighbour list potentially
        self._invocations += 1

        args2 = self._get_class_lib_args(cell2part)

        args = args2 + args

        # Execute the kernel over all particle pairs.
        method = self._lib[self._kernel.name + '_wrapper']

        self.wrapper_timer.start()
        method(*args)
        self.wrapper_timer.pause()

        self._update_opt()
        self._post_execute_dats(dat_dict)






