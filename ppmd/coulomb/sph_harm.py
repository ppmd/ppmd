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
        self.header = Include('math.h')

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


class _Symbol(object):
    def __init__(self, sym):
        self.sym = str(sym)
    def __str__(self):
        return self.sym
    def __add__(self, other):
        return _Symbol(str(self) + ' + ' + str(other))
    def __sub__(self, other):
        return _Symbol(str(self) + ' - ' + str(other))
    def __rsub__(self, other):
        return _Symbol(str(other) + ' - ' + str(self))    
    def __mul__(self, other):
        return _Symbol('' + str(self) + ' * ( ' + str(other) + ' )')
    __rmul__ = __mul__
    __radd__ = __add__


class SphExpGen(object):
    def __init__(self, maxl, esym='_E', psym='phi', ctype='double'):
        self.maxl = maxl
        self.psym = psym
        self.esym = esym
        self.ctype = ctype
        self.header = Include('math.h')

        cos_phi  = _Symbol('_cphi' + esym)
        sin_phi  = _Symbol('_sphi' + esym)
        msin_phi = _Symbol('_msphi' + esym)

        def icv_wrap(a, b):
            return Initializer(Const(Value(self.ctype, a)), b)

        modlist = [
            icv_wrap(self.get_e_sym(0)[0], '1.0'),
            icv_wrap(self.get_e_sym(0)[1], '0.0'),
            icv_wrap(sin_phi, 'sin({phi})'.format(phi=psym)),
            icv_wrap(cos_phi, 'cos({phi})'.format(phi=psym)),
            icv_wrap(msin_phi, -1.0 * sin_phi),
        ]

        for mx in range(1, maxl+1):
            re_m, im_m = self.get_e_sym(mx)
            re_mm1, im_mm1 = self.get_e_sym(mx-1)

            re_nm, im_nm = self.get_e_sym(-1 * mx)
            re_nmm1, im_nmm1 = self.get_e_sym(-1 * (mx - 1))
            
            re_m_rhs, im_m_rhs = self._cmplx_mul(cos_phi, sin_phi, re_mm1, im_mm1)
            re_nm_rhs, im_nm_rhs = self._cmplx_mul(cos_phi, msin_phi, re_nmm1, im_nmm1)

            modlist += [
                icv_wrap(re_m, re_m_rhs),
                icv_wrap(im_m, im_m_rhs),
                icv_wrap(re_nm, re_nm_rhs),
                icv_wrap(im_nm, im_nm_rhs)
            ]

        self.module = Module(modlist)

    @staticmethod
    def _cmplx_mul(a, b, x, y):
        g = a * x - b * y
        h = x * b + a * y
        return g, h
        
    def get_e_sym(self, m):
        assert abs(m) <= self.maxl
        s = self.esym
        s += 'm' if m < 0 else 'p'
        s += str(abs(m))
        return (_Symbol('_re' + s), _Symbol('_im' + s))




















