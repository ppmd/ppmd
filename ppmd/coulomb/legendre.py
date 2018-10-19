from __future__ import print_function, division, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

from cgen import *
from math import *

class ALegendrePolynomialGen(object):
    def __init__(self, maxl, psym='_P', tsym='theta', ctype='double'):
        self.maxl = maxl
        self.ctype = ctype
        self.psym = psym
        self.ptmp = psym + 'tmp'
        self.tmp_count = -1
        self.tsym = tsym
        
        sqrttmp = self._get_next_tmp()

        modlist = [
            Initializer(Const(Value(self.ctype, sqrttmp)), 'sqrt(1.0 - {theta} * {theta})'.format(theta=self.tsym)),
            Initializer(Const(Value(self.ctype, self.get_p_sym(0,0))), '1.0'),
        ]
        for lx in range(self.maxl):

            theta_lxp1 = self._get_next_tmp()
            modlist += [
                Initializer(Const(Value(self.ctype, theta_lxp1)),
                    '({theta})*({lx2p1})'.format(theta=self.tsym, lx2p1=str(float(2*lx + 1)))),
                Initializer(Const(Value(self.ctype, self.get_p_sym(lx+1,lx+1))),
                    '({m1m2lx})*({sqrttmp})*({plxlx})'.format(
                        m1m2lx=str(float(-1.0 - 2.0*lx)), sqrttmp=sqrttmp, plxlx=self.get_p_sym(lx, lx))
                ),
                Initializer(Const(Value(self.ctype, self.get_p_sym(lx+1,lx))),
                    '({theta_lxp1}) * ({plxlx})'.format(
                        theta=self.tsym, theta_lxp1=theta_lxp1, plxlx=self.get_p_sym(lx,lx))
                ),
            ]


            for mx in range(lx):
                modlist += [
                    Initializer(Const(Value(self.ctype, self.get_p_sym(lx+1, mx))),
                        '(({theta_lxp1}) * {plxmx} - ({lxpmx}) * {plx1mx} ) * ({ilxmmxp1})'.format(
                            theta_lxp1=theta_lxp1,
                            lx2p1=str(float(2*lx + 1)),
                            plxmx=self.get_p_sym(lx,mx),
                            plx1mx=self.get_p_sym(lx-1,mx),
                            ilxmmxp1=str(float(1.0/(lx - mx + 1))),
                            lxpmx=str(float(lx + mx))
                        )
                    ),                    
                ]
        self.module = Module(modlist)

    def get_p_sym(self, l, m):
        assert l > -1
        m = abs(m)
        assert m <= self.maxl
        assert l <= self.maxl
        return '{p}l{l}m{m}'.format(p=self.psym, l=l, m=m)
    
    def _get_next_tmp(self):
        self.tmp_count += 1
        return self.ptmp + str(self.tmp_count)



