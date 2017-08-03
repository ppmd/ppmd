import ctypes

import ppmd.modules.code_timer
from ppmd import host, opt
from ppmd.lib import build


def Restrict(keyword, symbol):
    return str(keyword) + ' ' + str(symbol)

class _Base(object):
    def __init__(self, n, kernel, dat_dict):

        self._cc = build.TMPCC

        self._N = n

        self._kernel = kernel

        self._dat_dict = dat_dict

        self.loop_timer = ppmd.modules.code_timer.LoopTimer()

        self._code_init()

        self._lib = build.simple_lib_creator(self._generate_header_source(),
                                             self._generate_impl_source(),
                                             self._kernel.name,
                                             CC=self._cc)
    def _code_init(self):
        pass

    def _generate_header_source(self):
        pass

    def _argnames(self):
        """Comma separated string of argument name declarations.

        This string of argument names is used in the declaration of
        the method which executes the pairloop over the grid.
        If, for example, the pairloop gets passed two particle_dats,
        then the result will be ``double** arg_000,double** arg_001`.`
        """


        argnames = str(self.loop_timer.get_cpp_arguments()) + ','


        if self._kernel.static_args is not None:
            self._static_arg_order = []

            for i, dat in enumerate(self._kernel.static_args.items()):
                argnames += 'const ' + host.ctypes_map[dat[1]] + ' ' + dat[0] + ','
                self._static_arg_order.append(dat[0])


        for i, dat in enumerate(self._dat_dict.items()):



            if type(dat[1]) is not tuple:
                argnames += host.ctypes_map[dat[1].dtype] + ' * ' + self._cc.restrict_keyword + ' ' + dat[0] + '_ext,'
            else:
                if not dat[1][1].write:
                    const_str = 'const '
                else:
                    const_str = ''
                argnames += const_str + host.ctypes_map[dat[1][0].dtype] + ' * ' + self._cc.restrict_keyword + ' ' + dat[0] + '_ext,'


        return argnames[:-1]

    def _included_headers(self):
        """Return names of included header files."""
        s = ''
        if self._kernel.headers is not None:
            s += '\n'
            for x in self._kernel.headers:
                s += '#include \"' + str(x) + '\" \n'

        s += str(self.loop_timer.get_cpp_headers())

        return s

    def _generate_impl_source(self):
        """Generate the source code the actual implementation.
        """

        d = {'KERNEL': self._kernel_code,
             'ARGUMENTS': self._argnames(),
             'KERNEL_NAME': self._kernel.name,
             'KERNEL_ARGUMENT_DECL': self._kernel_argument_declarations(),
             'LOOP_TIMER_PRE': str(self.loop_timer.get_cpp_pre_loop_code()),
             'LOOP_TIMER_POST': str(self.loop_timer.get_cpp_post_loop_code()),
             'RESTRICT': self._cc.restrict_keyword}

        return self._code % d

    def execute(self, n=None, dat_dict=None, static_args=None):

        """Allow alternative pointers"""
        if dat_dict is not None:
            self._dat_dict = dat_dict

        '''Currently assume n is always needed'''
        if n is not None:
            _N = n
        else:
            _N = self._N()

        args = [ctypes.c_int(_N)]

        args.append(self.loop_timer.get_python_parameters())


        '''TODO IMPLEMENT/CHECK RESISTANCE TO ARG REORDERING'''

        '''Add static arguments to launch command'''
        if self._kernel.static_args is not None:
            assert static_args is not None, "Error: static arguments not passed to loop."
            for dat in static_args.values():
                args.append(dat)

        '''Add pointer arguments to launch command'''
        for dat_orig in self._dat_dict.values():
            if type(dat_orig) is tuple:
                args.append(dat_orig[0].ctypes_data_access(dat_orig[1]), pair=True)
            else:
                raise RuntimeError
                #args.append(dat_orig.ctypes_data)

        '''Execute the kernel over all particle pairs.'''
        method = self._lib[self._kernel.name + '_wrapper']
        method(*args)

        '''after wards access descriptors'''
        for dat_orig in self._dat_dict.values():
            if type(dat_orig) is tuple:
                dat_orig[0].ctypes_data_post(dat_orig[1])