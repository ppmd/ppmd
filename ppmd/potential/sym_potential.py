__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

from ppmd.kernel import Kernel, Constant, Header
import ppmd.access as access

import pymbolic as pmbl
from pymbolic.mapper.c_code import CCodeMapper as CCM
from pymbolic.mapper.evaluator import EvaluationMapper as EM

class RadiusSquared(object):
    def __new__(self):
        return pmbl.var('_r2')


class SymbolicPotential(object):
    def __init__(self, rc, potential_expr, force_expr=None, shift=True):

        self._rc = rc
        if shift:
            potential_expr -= EM(context={"_r2": self._rc**2.})(potential_expr)
        self.potential_expr = potential_expr
        self.force_expr = force_expr

    @property
    def kernel(self):
        """
        Returns a kernel class for the potential.
        """

        ccm = CCM()

        UEXPR = ccm(0.5*self.potential_expr)
        FEXPR = ccm(self.force_expr)

        kernel_code = '''
        const double R0 = P.j[0] - P.i[0];
        const double R1 = P.j[1] - P.i[1];
        const double R2 = P.j[2] - P.i[2];

        const double _r2 = R0*R0 + R1*R1 + R2*R2;

        u[0]+= (_r2 < _rc2) ? %(UEXPR)s : 0.0;

        const double f_tmp = %(FEXPR)s;

        A.i[0]-= (_r2 < _rc2) ? f_tmp*R0 : 0.0;
        A.i[1]-= (_r2 < _rc2) ? f_tmp*R1 : 0.0;
        A.i[2]-= (_r2 < _rc2) ? f_tmp*R2 : 0.0;

        ''' % {'UEXPR': UEXPR, 'FEXPR': FEXPR}

        constants = (Constant('_rc2', self._rc ** 2),)

        return Kernel('SymbolicPotential', kernel_code, constants, [Header('stdio.h')])


    def get_data_map(self, positions=None, forces=None, potential_energy=None):
         return {'P': positions(access.R), 'A': forces(access.INC0), 'u': potential_energy(access.INC0)}






