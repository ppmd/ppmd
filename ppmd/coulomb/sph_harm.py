from __future__ import print_function, division, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

from cgen import *
from math import *

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

class ALegendrePolynomialGen(object):
    def __init__(self, maxl, psym='_P', tsym='theta', ctype='double', avoid_calls=False):
        self.maxl = maxl
        self.ctype = ctype
        self.psym = psym
        self.ptmp = psym + 'tmp'
        self.tmp_count = -1
        self.tsym = tsym
        self.header = Include('math.h')
        self.flops = {'+': 0, '-': 0, '*': 0, '/': 0}

        modlist = []
        if not avoid_calls:
            sqrttmp = self._get_next_tmp()
            sqrt_theta_sym = 'sqrt(1.0 - {theta} * {theta})'.format(theta=self.tsym)
            modlist += [
                Initializer(Const(Value(self.ctype, sqrttmp)), sqrt_theta_sym),
            ]
        else:
            sqrttmp = 'sqrt_theta_tmp'


        modlist += [
            Initializer(Const(Value(self.ctype, self.get_p_sym(0,0))), '1.0'),
        ]


        for lx in range(self.maxl):

            theta_lxp1 = self._get_next_tmp()
            modlist += [
                Initializer(Const(Value(self.ctype, theta_lxp1)),
                    '({theta})*({lx2p1})'.format(theta=self.tsym, lx2p1=str(float(2*lx + 1)))),
                Initializer(Const(Value(self.ctype, str(self.get_p_sym(lx+1,lx+1)))),
                    '({m1m2lx})*({sqrttmp})*({plxlx})'.format(
                        m1m2lx=str(float(-1.0 - 2.0*lx)), sqrttmp=sqrttmp, plxlx=str(self.get_p_sym(lx, lx)))
                ),
                Initializer(Const(Value(self.ctype, self.get_p_sym(lx+1,lx))),
                    '({theta_lxp1}) * ({plxlx})'.format(
                        theta=self.tsym, theta_lxp1=theta_lxp1, plxlx=str(self.get_p_sym(lx,lx)))
                ),
            ]
            self.flops['*'] += 3

            for mx in range(lx):
                modlist += [
                    Initializer(Const(Value(self.ctype, self.get_p_sym(lx+1, mx))),
                        '(({theta_lxp1}) * {plxmx} - ({lxpmx}) * {plx1mx} ) * ({ilxmmxp1})'.format(
                            theta_lxp1=theta_lxp1,
                            lx2p1=str(float(2*lx + 1)),
                            plxmx=str(self.get_p_sym(lx,mx)),
                            plx1mx=str(self.get_p_sym(lx-1,mx)),
                            ilxmmxp1=str(float(1.0/(lx - mx + 1))),
                            lxpmx=str(float(lx + mx))
                        )
                    ),                    
                ]
                self.flops['*'] += 4
                self.flops['-'] += 1

        self.module = Module(modlist)

    def get_p_sym(self, l, m):
        assert l > -1
        m = abs(m)
        assert m <= self.maxl
        assert l <= self.maxl
        return _Symbol('{p}l{l}m{m}'.format(p=self.psym, l=l, m=m))
    
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
    def __init__(self, maxl, esym='_E', psym='phi', ctype='double', avoid_calls=False):
        self.maxl = maxl
        self.psym = psym
        self.esym = esym
        self.ctype = ctype
        self.header = Include('math.h')
        self.flops = {'+': 0, '-': 0, '*': 0, '/': 0}

        cos_phi  = _Symbol('_cphi' + esym)
        sin_phi  = _Symbol('_sphi' + esym)
        msin_phi = _Symbol('_msphi' + esym)

        def icv_wrap(a, b):
            return Initializer(Const(Value(self.ctype, a)), b)

        modlist = []
        if not avoid_calls:
            modlist += [
                icv_wrap(sin_phi, 'sin({phi})'.format(phi=psym)),
                icv_wrap(cos_phi, 'cos({phi})'.format(phi=psym)),           
            ]
        else:
            modlist += [
                icv_wrap(sin_phi, 'sin_phi'),
                icv_wrap(cos_phi, 'cos_phi'),           
            ]

        modlist += [
            icv_wrap(self.get_e_sym(0)[0], '1.0'),
            icv_wrap(self.get_e_sym(0)[1], '0.0'),
            icv_wrap(msin_phi, -1.0 * sin_phi),
        ]
        self.flops['*'] += 1

        for mx in range(1, maxl+1):
            re_m, im_m = self.get_e_sym(mx)
            re_mm1, im_mm1 = self.get_e_sym(mx-1)

            re_nm, im_nm = self.get_e_sym(-1 * mx)
            re_nmm1, im_nmm1 = self.get_e_sym(-1 * (mx - 1))
            
            re_m_rhs, im_m_rhs = cmplx_mul(cos_phi, sin_phi, re_mm1, im_mm1)
            re_nm_rhs, im_nm_rhs = cmplx_mul(cos_phi, msin_phi, re_nmm1, im_nmm1)

            self.flops['*'] += 8
            self.flops['+'] += 2
            self.flops['-'] += 2

            modlist += [
                icv_wrap(re_m, re_m_rhs),
                icv_wrap(im_m, im_m_rhs),
                icv_wrap(re_nm, re_nm_rhs),
                icv_wrap(im_nm, im_nm_rhs)
            ]

        self.module = Module(modlist)
        
    def get_e_sym(self, m):
        assert abs(m) <= self.maxl
        s = self.esym
        s += 'm' if m < 0 else 'p'
        s += str(abs(m))
        return (_Symbol('_re' + s), _Symbol('_im' + s))

SphSymbol = _Symbol

def cmplx_mul(a, b, x, y):
    g = a * x - b * y
    h = x * b + a * y
    return g, h


class SphCoeffGen(object):
    def __init__(self, maxl, sym='_sqrtmf', ctype='double'):
        self.maxl = maxl
        self.sym = sym
        self.ctype = ctype
        self.header = ''
        self.flops = {'+': 0, '-': 0, '*': 0, '/': 0}

        def icv_wrap(a, b):
            return Initializer(Const(Value(self.ctype, a)), b)
        
        modlist = [icv_wrap(self.get_numerator_sym(lx), sqrt(factorial(lx))) for lx in range(maxl+1)]
        modlist += [icv_wrap(self.get_denominator_sym(lx), 1.0 / sqrt(factorial(lx))) for lx in range(2*maxl+1)]

        self.module = Module(modlist)
    
    def get_numerator_sym(self, mx):
        assert mx > -1
        assert abs(mx) <= self.maxl
        return _Symbol(self.sym + '_num_' + str(mx))
    
    def get_denominator_sym(self, mx):
        assert mx > -1
        assert abs(mx) <= 2*self.maxl+1
        return _Symbol(self.sym + '_denom_' + str(mx))


class SphGen(object):
    def __init__(self, maxl, sym='_Y', theta_sym='theta', phi_sym='phi', ctype='double', avoid_calls=False):
        self.maxl = maxl
        self.sym = sym
        self.ctype = ctype
        self.header = Include('math.h')
        self.flops = {'+': 0, '-': 0, '*': 0, '/': 0}

        def icv_wrap(a, b):
            return Initializer(Const(Value(self.ctype, a)), b)
 
        modlist = []
        if avoid_calls:
           pass
        else:
            modlist += [
                icv_wrap('cos_' + theta_sym, 'cos(' + theta_sym + ')'),
            ]

        lpmv_gen = ALegendrePolynomialGen(maxl=maxl, tsym='cos_' + theta_sym, ctype=ctype,
                avoid_calls=avoid_calls)

        exp_gen = SphExpGen(maxl=maxl, psym=phi_sym, ctype=ctype, avoid_calls=avoid_calls)
        coeff_gen = SphCoeffGen(maxl=maxl, ctype=ctype)
        

        modlist += [
            lpmv_gen.module,
            exp_gen.module,
            coeff_gen.module
        ]
        
        for kx in self.flops.keys():
            self.flops[kx] += lpmv_gen.flops[kx] + exp_gen.flops[kx] + coeff_gen.flops[kx]
        
        for lx in range(maxl+1):
            modlist += [
                icv_wrap(
                    self.get_y_sym(lx, mx)[0],
                    coeff_gen.get_numerator_sym(lx - abs(mx)) * coeff_gen.get_denominator_sym(lx + abs(mx)) * \
                        lpmv_gen.get_p_sym(lx, abs(mx)) * exp_gen.get_e_sym(mx)[0]
                ) for mx in range(-lx, lx+1)
            ]
            modlist += [
                icv_wrap(
                    self.get_y_sym(lx, mx)[1],
                    coeff_gen.get_numerator_sym(lx - abs(mx)) * coeff_gen.get_denominator_sym(lx + abs(mx)) * \
                        lpmv_gen.get_p_sym(lx, abs(mx)) * exp_gen.get_e_sym(mx)[1]
                ) for mx in range(-lx, lx+1)
            ]

            self.flops['*'] += 6 * (2*lx+1)

        self.module = Module(modlist)

    def get_y_sym(self, n, m):
        assert n > -1
        assert n <= self.maxl
        assert abs(m) <= n
        ms = 'mn' if m < 0 else 'mp'
        ms += str(abs(m))
        s = self.sym + 'n' + str(abs(n)) + ms
        return _Symbol('_re' + s), _Symbol('_im' + s)











