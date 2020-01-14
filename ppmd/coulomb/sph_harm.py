from __future__ import print_function, division, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

from cgen import *
from math import *
from ctypes import c_double, c_int, byref

import numpy as np
import scipy
from scipy.special import lpmv

from functools import lru_cache
from ppmd.lib.build import simple_lib_creator



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



class PRadiusModifier:
    def __init__(self, lpmv_gen, radius_symbol=False):
        
        self.lpmv_gen = lpmv_gen
        self.radius_symbol = radius_symbol
        modules = []
        
        if self.radius_symbol:
            maxl = lpmv_gen.maxl
            for lx in range(maxl + 1):
                for mx in range(lx+1):
                    modules += [self.get_line(lx, mx)]

        self.module = Module(modules)

    def get_p_sym(self, l, m):
        if not self.radius_symbol:
            return self.lpmv_gen.get_p_sym(l, m)
        else:
            return _Symbol('rc_{}'.format(self.lpmv_gen.get_p_sym(l, m)))

    def get_radius_sym(self, l):
        assert self.radius_symbol
        return _Symbol(self.radius_symbol+'_'+str(l))

    def get_line(self, lx, mx):

        if self.radius_symbol:
            return Initializer(
                Const(
                    Value(self.lpmv_gen.ctype, self.get_p_sym(lx, mx))
                ), 
                self.lpmv_gen.get_p_sym(lx, mx) * self.get_radius_sym(lx)
            )
        else:
            return Comment('No radius coefficient for l={} m={}'.format(lx, mx))


class SphGen(object):
    def __init__(self, maxl, sym='_Y', theta_sym='theta', phi_sym='phi', ctype='double', avoid_calls=False, radius_symbol=False):
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

        lpmv_gen = ALegendrePolynomialGen(maxl=maxl, psym='_P'+sym, tsym='cos_' + theta_sym, ctype=ctype,
                avoid_calls=avoid_calls)

        radius_lpmv_gen = PRadiusModifier(lpmv_gen, radius_symbol=radius_symbol)

        exp_gen = SphExpGen(maxl=maxl, esym='_E'+sym, psym=phi_sym, ctype=ctype, avoid_calls=avoid_calls)
        #coeff_gen = SphCoeffGen(maxl=maxl, sym='_sqrtmf'+sym, ctype=ctype)
        

        modlist += [
            lpmv_gen.module,
            radius_lpmv_gen.module,
            exp_gen.module,
            #coeff_gen.module
        ]
        
        for kx in self.flops.keys():
            self.flops[kx] += lpmv_gen.flops[kx] + exp_gen.flops[kx]# + coeff_gen.flops[kx]

        
        def H(lx, mx):
            return _Symbol(str(sqrt(factorial(lx - abs(mx)) / factorial(lx + abs(mx)))))


        for lx in range(maxl+1):
            modlist += [
                icv_wrap(
                    self._get_intermediate_sym(lx, mx),
                    H(lx, mx) * radius_lpmv_gen.get_p_sym(lx, abs(mx))
                ) for mx in range(0, lx+1)
            ]
            self.flops['*'] += lx+1


        
        for lx in range(maxl+1):
            modlist += [
                icv_wrap(
                    self.get_y_sym(lx, mx)[0],
                    self._get_intermediate_sym(lx, abs(mx)) * exp_gen.get_e_sym(mx)[0]
                ) for mx in range(-lx, lx+1)
            ]
            modlist += [
                icv_wrap(
                    self.get_y_sym(lx, mx)[1],
                    self._get_intermediate_sym(lx, abs(mx)) * exp_gen.get_e_sym(mx)[1]
                ) for mx in range(-lx, lx+1)
            ]

            self.flops['*'] += 2 * (2*lx+1)



        self.module = Module(modlist)
        self.radius_lpmv_gen = radius_lpmv_gen

    def get_y_sym(self, n, m):
        assert n > -1
        assert n <= self.maxl
        assert abs(m) <= n
        ms = 'mn' if m < 0 else 'mp'
        ms += str(abs(m))
        s = self.sym + 'n' + str(abs(n)) + ms
        return _Symbol('_re' + s), _Symbol('_im' + s)

    def _get_intermediate_sym(self, n, m):
        assert n > -1
        assert n <= self.maxl
        assert abs(m) <= n
        ms = 'mn' if m < 0 else 'mp'
        ms += str(abs(m))
        s = self.sym + 'n' + str(abs(n)) + ms
        return _Symbol('_Hnm_' + s)

    def get_radius_sym(self, n):
        return self.radius_lpmv_gen.get_radius_sym(n)


class LocalExpEval(object):
    
    def __init__(self, L):
        self.L = L
        self._hmatrix_py = np.zeros((2*self.L, 2*self.L))
        def Hfoo(nx, mx): return sqrt(float(factorial(nx - abs(mx)))/factorial(nx + abs(mx)))
        for nx in range(self.L):
            for mx in range(-nx, nx+1):
                self._hmatrix_py[nx, mx] = Hfoo(nx, mx)
        
        self.sph_gen = SphGen(L-1)
        sph_gen = self.sph_gen

        def cube_ind(L, M):
            return ((L) * ( (L) + 1 ) + (M) )

        assign_gen = ''
        for lx in range(self.L):
            for mx in range(-lx, lx+1):
                reL = SphSymbol('moments[{ind}]'.format(ind=cube_ind(lx, mx)))
                imL = SphSymbol('moments[IM_OFFSET + {ind}]'.format(ind=cube_ind(lx, mx)))
                reY, imY = sph_gen.get_y_sym(lx, mx)
                phi_sym = cmplx_mul(reL, imL, reY, imY)[0]
                assign_gen += 'tmp_energy += rhol * ({phi_sym});\n'.format(phi_sym=str(phi_sym))

            assign_gen += 'rhol *= radius;\n'

        src = """
        #define IM_OFFSET ({IM_OFFSET})

        {DECLARE} int local_eval(
            const double radius,
            const double theta,
            const double phi,
            const double * RESTRICT moments,
            double * RESTRICT out
        ){{
            {SPH_GEN}
            double rhol = 1.0;
            double tmp_energy = 0.0;
            {ASSIGN_GEN}

            out[0] = tmp_energy;
            return 0;
        }}
        """
        
        src_lib = src.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
            DECLARE=r'static inline'
        )

        src = src.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
            DECLARE=r'extern "C"'
        )
        header = str(sph_gen.header)


        self.create_local_eval_header = header
        self.create_local_eval_src = src_lib

        self._local_eval_lib = simple_lib_creator(header_code=header, src_code=src)['local_eval']

    

    def __call__(self, moments, disp_sph):
        assert moments.dtype == c_double
        _out = c_double(0.0)
        self._local_eval_lib(
            c_double(disp_sph[0]),
            c_double(disp_sph[1]),
            c_double(disp_sph[2]),
            moments.ctypes.get_as_parameter(),
            byref(_out)
        )
        return _out.value


    def py_compute_phi_local(self, moments, disp_sph):
        """
        Computes the field at the podint disp_sph given by the local expansion 
        in moments
        """

        llimit = self.L
    
        phi_sph_re = 0.
        phi_sph_im = 0.
        def re_lm(l,m): return (l**2) + l + m
        def im_lm(l,m): return (l**2) + l +  m + llimit**2

        cosv = np.zeros(3 * llimit)
        sinv = np.zeros(3 * llimit)
        for mx in range(-llimit, llimit+1):
            cosv[mx] = cos(mx * disp_sph[2])
            sinv[mx] = sin(mx * disp_sph[2])

        for lx in range(llimit):
            scipy_p = lpmv(range(lx+1), lx, np.cos(disp_sph[1]))
            irad = disp_sph[0] ** (lx)
            for mx in range(-lx, lx+1):

                val = self._hmatrix_py[lx, mx] * scipy_p[abs(mx)]

                scipy_real = cosv[mx] * val * irad
                scipy_imag = sinv[mx] * val * irad

                ppmd_mom_re = moments[re_lm(lx, mx)]
                ppmd_mom_im = moments[im_lm(lx, mx)]

                phi_sph_re += scipy_real*ppmd_mom_re - scipy_imag*ppmd_mom_im

        return phi_sph_re


class MultipoleExpCreator:
    """
    Class to compute multipole expansions.

    :arg int L: Number of expansion terms.
    """
    
    def __init__(self, L):
        self.L = L
        self._hmatrix_py = np.zeros((2*self.L, 2*self.L))
        def Hfoo(nx, mx): return sqrt(float(factorial(nx - abs(mx)))/factorial(nx + abs(mx)))
        for nx in range(self.L):
            for mx in range(-nx, nx+1):
                self._hmatrix_py[nx, mx] = Hfoo(nx, mx)
        
        self.sph_gen = SphGen(L-1)
        self._multipole_lib = None
        self._generate_host_libs()

    def _generate_host_libs(self):

        sph_gen = self.sph_gen

        def cube_ind(L, M):
            return ((L) * ( (L) + 1 ) + (M) )

        assign_gen =  'double rhol = 1.0;\n'
        assign_gen += 'double rholcharge = rhol * charge;\n'
        for lx in range(self.L):
            for mx in range(-lx, lx+1):
                assign_gen += 'out[{ind}] += {ylmm} * rholcharge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, -mx)[0])
                    )
                assign_gen += 'out[IM_OFFSET + {ind}] += {ylmm} * rholcharge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, -mx)[1])
                    )
            assign_gen += 'rhol *= radius;\n'
            assign_gen += 'rholcharge = rhol * charge;\n'

        src = """
        #define IM_OFFSET ({IM_OFFSET})

        {DECLARE} int multipole_exp(
            const double charge,
            const double radius,
            const double theta,
            const double phi,
            double * RESTRICT out
        ){{
            {SPH_GEN}
            {ASSIGN_GEN}
            return 0;
        }}
        """
        header = str(sph_gen.header)
        
        src_lib = src.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
            DECLARE=r'static inline'
        )

        src = src.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
            DECLARE=r'extern "C"'
        )

        self.create_multipole_header = header
        self.create_multipole_src = src_lib

        self._multipole_lib = simple_lib_creator(header_code=header, src_code=src)['multipole_exp']


    def multipole_exp(self, sph, charge, arr):
        """
        For a charge at the point sph computes the multipole moments at the origin
        and appends them onto arr.
        """

        assert arr.dtype == c_double
        self._multipole_lib(
            c_double(charge),
            c_double(sph[0]),
            c_double(sph[1]),
            c_double(sph[2]),
            arr.ctypes.get_as_parameter()
        )


    def py_multipole_exp(self, sph, charge, arr):
        """
        For a charge at the point sph computes the multipole moments at the origin
        and appends them onto arr.
        """

        llimit = self.L
        def re_lm(l,m): return (l**2) + l + m
        def im_lm(l,m): return (l**2) + l +  m + llimit**2
        
        cosv = np.zeros(3 * llimit)
        sinv = np.zeros(3 * llimit)
        for mx in range(-llimit, llimit+1):
            cosv[mx] = cos(-1.0 * mx * sph[2])
            sinv[mx] = sin(-1.0 * mx * sph[2])

        for lx in range(self.L):
            scipy_p = lpmv(range(lx+1), lx, cos(sph[1]))
            radn = sph[0] ** lx
            for mx in range(-lx, lx+1):
                coeff = charge * radn * self._hmatrix_py[lx, mx] * scipy_p[abs(mx)] 
                arr[re_lm(lx, mx)] += cosv[mx] * coeff
                arr[im_lm(lx, mx)] += sinv[mx] * coeff



class MultipoleDotVecCreator:
    """
    Class to compute multipole expansions and corresponding expansions that can be used to compute
    energies from local expansions by using a dot product.

    :arg int L: Number of expansion terms.
    """
    
    def __init__(self, L):
        self.L = L
        self._hmatrix_py = np.zeros((2*self.L, 2*self.L))
        def Hfoo(nx, mx): return sqrt(float(factorial(nx - abs(mx)))/factorial(nx + abs(mx)))
        for nx in range(self.L):
            for mx in range(-nx, nx+1):
                self._hmatrix_py[nx, mx] = Hfoo(nx, mx)
        
        self.sph_gen = SphGen(L-1)
        self._multipole_lib = None
        self._generate_host_libs()

    def _generate_host_libs(self):

        sph_gen = self.sph_gen

        def cube_ind(L, M):
            return ((L) * ( (L) + 1 ) + (M) )

        # --- lib to create vector to dot product and mutlipole expansions --- 

        assign_gen =  'double rhol = 1.0;\n'
        assign_gen += 'double rholcharge = rhol * charge;\n'
        flops = {'+': 0, '-': 0, '*': 0, '/': 0}

        for lx in range(self.L):
            for mx in range(-lx, lx+1):

                assign_gen += 'out_mul[{ind}] += {ylmm} * rholcharge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, -mx)[0])
                    )
                assign_gen += 'out_mul[IM_OFFSET + {ind}] += {ylmm} * rholcharge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, -mx)[1])
                    )
                assign_gen += 'out_vec[{ind}] += {ylmm} * rholcharge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, mx)[0])
                    )
                assign_gen += 'out_vec[IM_OFFSET + {ind}] += (-1.0) * {ylmm} * rholcharge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, mx)[1])
                    )

                flops['+'] += 4
                flops['*'] += 5

            assign_gen += 'rhol *= radius;\n'
            assign_gen += 'rholcharge = rhol * charge;\n'
            flops['*'] += 2

        flops['+'] += sph_gen.flops['*']
        flops['-'] += sph_gen.flops['*']
        flops['*'] += sph_gen.flops['*']
        flops['/'] += sph_gen.flops['*']

        src = """
        #define IM_OFFSET ({IM_OFFSET})

        {DECLARE} int local_dot_vec_multipole(
            const double charge,
            const double radius,
            const double theta,
            const double phi,
            double * RESTRICT out_vec,
            double * RESTRICT out_mul
        ){{
            {SPH_GEN}
            {ASSIGN_GEN}
            return 0;
        }}
        """
        header = str(sph_gen.header)
        
        src_lib = src.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
            DECLARE='static inline'
        )

        src = src.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
            DECLARE=r'extern "C"'
        )

        self.create_dot_vec_multipole_header = header
        self.create_dot_vec_multipole_src = src_lib
        self.create_dot_vec_multipole_flops = flops

        self._dot_vec_multipole_lib = simple_lib_creator(header_code=header, src_code=src)['local_dot_vec_multipole']
        

    def dot_vec_multipole(self, sph, charge, arr_vec, arr_mul):
        """
        For a charge at the point sph computes the coefficients at the origin
        and appends them onto arr that can be used in a dot product to compute
        the energy.
        """
        assert arr_vec.dtype == c_double
        assert arr_mul.dtype == c_double
        self._dot_vec_multipole_lib(
            c_double(charge),
            c_double(sph[0]),
            c_double(sph[1]),
            c_double(sph[2]),
            arr_vec.ctypes.get_as_parameter(),
            arr_mul.ctypes.get_as_parameter()
       ) 


    def py_dot_vec(self, sph, charge, arr):
        """
        For a charge at the point sph computes the multipole moments at the origin
        and appends them onto arr.
        """

        llimit = self.L
        def re_lm(l,m): return (l**2) + l + m
        def im_lm(l,m): return (l**2) + l +  m + llimit**2
        
        cosv = np.zeros(3 * llimit)
        sinv = np.zeros(3 * llimit)
        for mx in range(-llimit, llimit+1):
            cosv[mx] = cos(mx * sph[2])
            sinv[mx] = sin(mx * sph[2])

        for lx in range(self.L):
            scipy_p = lpmv(range(lx+1), lx, cos(sph[1]))
            radn = sph[0] ** lx
            for mx in range(-lx, lx+1):
                coeff = charge * radn * self._hmatrix_py[lx, mx] * scipy_p[abs(mx)] 
                arr[re_lm(lx, mx)] += cosv[mx] * coeff
                arr[im_lm(lx, mx)] -= sinv[mx] * coeff       



























class ALegendrePolynomialGenEphemeral(object):
    def __init__(self, maxl, psym='_P', tsym='theta', ctype='double', avoid_calls=False):
        self.maxl = maxl
        self.ctype = ctype
        self.avoid_calls = avoid_calls
        self.psym = psym
        self.ptmp = psym + 'tmp'
        self.tmp_count = -1
        self.tsym = tsym
        self.header = Include('math.h')
        self.flops = {'+': 0, '-': 0, '*': 0, '/': 0}
    
    def __call__(self, d):

        modlist = []
        if not self.avoid_calls:
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
        


        basetmp = self._get_next_tmp()
        theta_lxp1 = [basetmp + '_{}'.format(str(lx)) for lx in range(self.maxl+1)]
        

        modlist += [
            Initializer(Const(Value(self.ctype, theta_lxp1[lx])),
                '({theta})*({lx2p1})'.format(theta=self.tsym, lx2p1=str(float(2*lx + 1)))) for lx in range(0, self.maxl+1)
        ]

        for mx in range(0, self.maxl):
            modlist += [
                Initializer(Const(Value(self.ctype, self.get_p_sym(mx+1, mx))),
                    '({theta_lxp1}) * ({plxlx})'.format(
                        theta=self.tsym, theta_lxp1=theta_lxp1[mx], plxlx=str(self.get_p_sym(mx, mx)))
                )
            ]
            if (mx, mx) in d.keys():
                modlist += [
                    Line(str(dx)) for dx in d[(mx, mx)] 
                ]
            if (mx+1, mx) in d.keys():
                modlist += [
                    Line(str(dx)) for dx in d[(mx+1, mx)] 
                ]

            for lx in range(mx+1, self.maxl):
                modlist += [
                    Initializer(Const(Value(self.ctype, self.get_p_sym(lx+1, mx))),
                        '(({theta_lxp1}) * {plxmx} - ({lxpmx}) * {plx1mx} ) * ({ilxmmxp1})'.format(
                            theta_lxp1=theta_lxp1[lx],
                            lx2p1=str(float(2*lx + 1)),
                            plxmx=str(self.get_p_sym(lx,mx)),
                            plx1mx=str(self.get_p_sym(lx-1,mx)),
                            ilxmmxp1=str(float(1.0/(lx - mx + 1))),
                            lxpmx=str(float(lx + mx))
                        )
                    )
                ]
                if (lx+1, mx) in d.keys():
                    modlist += [
                        Line(str(dx)) for dx in d[(lx+1, mx)] 
                    ]


            modlist += [
                Initializer(Const(Value(self.ctype, str(self.get_p_sym(mx+1,mx+1)))),
                    '({m1m2lx})*({sqrttmp})*({plxlx})'.format(
                        m1m2lx=str(float(-1.0 - 2.0*mx)), sqrttmp=sqrttmp, plxlx=str(self.get_p_sym(mx, mx)))
                )
            ]

        if (self.maxl, self.maxl) in d.keys():
            modlist += [
                Line(str(dx)) for dx in d[(self.maxl, self.maxl)] 
            ]



        """
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

            if (lx, 0) in d.keys():
                modlist += [
                    Line(str(dx)) for dx in d[(lx, 0)] 
                ]

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
                    )
                ]
                self.flops['*'] += 4
                self.flops['-'] += 1

                if (lx, mx+1) in d.keys():
                    modlist += [
                        Line(str(dx)) for dx in d[(lx, mx+1)] 
                    ]


        lx = self.maxl
        for mx in range(0, lx+1):
            if (lx, mx) in d.keys():
                modlist += [
                    Line(str(dx)) for dx in d[(lx, mx)] 
                ]
        """



        self.module = Module(modlist)
        return modlist
        

    def get_p_sym(self, l, m):
        assert l > -1
        m = abs(m)
        assert m <= self.maxl
        assert l <= self.maxl
        return _Symbol('{p}l{l}m{m}'.format(p=self.psym, l=l, m=m))
    
    def _get_next_tmp(self):
        self.tmp_count += 1
        return self.ptmp + str(self.tmp_count)











class SphGenEphemeral(object):
    def __init__(self, maxl, sym='_Y', theta_sym='theta', phi_sym='phi', ctype='double', avoid_calls=False, radius_symbol=False):
        self.maxl = maxl
        self.sym = sym
        self.theta_sym = theta_sym
        self.phi_sym = phi_sym
        self.ctype = ctype
        self.avoid_calls = avoid_calls
        self.radius_symbol = radius_symbol
        self.header = Include('math.h')
        self.flops = {'+': 0, '-': 0, '*': 0, '/': 0}
        self.lpmv_gen = ALegendrePolynomialGenEphemeral(maxl=self.maxl, psym='_P'+sym, tsym='cos_' + theta_sym, ctype=self.ctype,
                avoid_calls=self.avoid_calls)

        radius_lpmv_gen = PRadiusModifier(self.lpmv_gen, radius_symbol=radius_symbol)
        self.radius_lpmv_gen = radius_lpmv_gen



    def __call__(self, din):

        def icv_wrap(a, b):
            return Initializer(Const(Value(self.ctype, a)), b)
 
        modlist = []
        if self.avoid_calls:
           pass
        else:
            modlist += [
                icv_wrap('cos_' + self.theta_sym, 'cos(' + self.theta_sym + ')'),
            ]



        exp_gen = SphExpGen(maxl=self.maxl, esym='_E'+self.sym, psym=self.phi_sym, ctype=self.ctype, avoid_calls=self.avoid_calls)
        #coeff_gen = SphCoeffGen(maxl=maxl, sym='_sqrtmf'+sym, ctype=self.ctype)
        

        modlist += [
            exp_gen.module,
        ]
        
        for kx in self.flops.keys():
            self.flops[kx] += self.lpmv_gen.flops[kx] + exp_gen.flops[kx]# + coeff_gen.flops[kx]

        
        def H(lx, mx):
            return _Symbol(str(sqrt(factorial(lx - abs(mx)) / factorial(lx + abs(mx)))))

        
        d = {}

        for lx in range(self.maxl+1):
            self.flops['*'] += lx+1
            self.flops['*'] += 2 * (2*lx+1)

            mx = 0

            ml = [
                self.radius_lpmv_gen.get_line(lx, mx),
                icv_wrap(
                    self._get_intermediate_sym(lx, mx),
                    H(lx, mx) * self.radius_lpmv_gen.get_p_sym(lx, abs(mx))
                ),
                icv_wrap(
                    self.get_y_sym(lx, mx)[0],
                    self._get_intermediate_sym(lx, abs(mx)) * exp_gen.get_e_sym(mx)[0]
                ),
                icv_wrap(
                    self.get_y_sym(lx, mx)[1],
                    self._get_intermediate_sym(lx, abs(mx)) * exp_gen.get_e_sym(mx)[1]
                )
            ]
            if (lx, mx) in din.keys():
                ml += [
                    Line(str(dx)) for dx in din[(lx, mx)] 
                ]

            d[(lx, mx)] = ml

            for mx in range(1, lx+1):
                ml = [
                    self.radius_lpmv_gen.get_line(lx, mx),
                    icv_wrap(
                        self._get_intermediate_sym(lx, mx),
                        H(lx, mx) * self.radius_lpmv_gen.get_p_sym(lx, abs(mx))
                    ),
                    icv_wrap(
                        self.get_y_sym(lx, mx)[0],
                        self._get_intermediate_sym(lx, abs(mx)) * exp_gen.get_e_sym(mx)[0]
                    ),
                    icv_wrap(
                        self.get_y_sym(lx, mx)[1],
                        self._get_intermediate_sym(lx, abs(mx)) * exp_gen.get_e_sym(mx)[1]
                    ),
                    icv_wrap(
                        self.get_y_sym(lx, -mx)[0],
                        self._get_intermediate_sym(lx, abs(mx)) * exp_gen.get_e_sym(-mx)[0]
                    ),
                    icv_wrap(
                        self.get_y_sym(lx, -mx)[1],
                        self._get_intermediate_sym(lx, abs(mx)) * exp_gen.get_e_sym(-mx)[1]
                    )
                ]
                if (lx, mx) in din.keys():
                    ml += [
                        Line(str(dx)) for dx in din[(lx, mx)] 
                    ]
                if (lx, -mx) in din.keys():
                    ml += [
                        Line(str(dx)) for dx in din[(lx, -mx)] 
                    ]


                d[(lx, mx)] = ml


        modlist += self.lpmv_gen(d)


        self.module = Module(modlist)

        return self.module





    def get_y_sym(self, n, m):
        assert n > -1
        assert n <= self.maxl
        assert abs(m) <= n
        ms = 'mn' if m < 0 else 'mp'
        ms += str(abs(m))
        s = self.sym + 'n' + str(abs(n)) + ms
        return _Symbol('_re' + s), _Symbol('_im' + s)


    def _get_intermediate_sym(self, n, m):
        assert n > -1
        assert n <= self.maxl
        assert abs(m) <= n
        ms = 'mn' if m < 0 else 'mp'
        ms += str(abs(m))
        s = self.sym + 'n' + str(abs(n)) + ms
        return _Symbol('_Hnm_' + s)


    def get_radius_sym(self, n):
        return self.radius_lpmv_gen.get_radius_sym(n)



def py_multipole_exp(L, sph, charge, arr):
    """
    For a charge at the point sph computes the multipole moments at the origin
    and appends them onto arr.
    """

    llimit = L
    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + llimit**2
    def Hfoo(nx, mx): return sqrt(float(factorial(nx - abs(mx)))/factorial(nx + abs(mx)))
    
    cosv = np.zeros(3 * llimit)
    sinv = np.zeros(3 * llimit)
    for mx in range(-llimit, llimit+1):
        cosv[mx] = cos(-1.0 * mx * sph[2])
        sinv[mx] = sin(-1.0 * mx * sph[2])

    for lx in range(L):
        scipy_p = lpmv(range(lx+1), lx, cos(sph[1]))
        radn = sph[0] ** lx
        for mx in range(-lx, lx+1):
            coeff = charge * radn * Hfoo(lx, mx) * scipy_p[abs(mx)] 
            arr[re_lm(lx, mx)] += cosv[mx] * coeff
            arr[im_lm(lx, mx)] += sinv[mx] * coeff


def py_local_exp(L, sph, charge, arr):
    """
    For a charge at the point sph computes the local moments at the origin
    and appends them onto arr.
    """

    llimit = L
    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + llimit**2
    def Hfoo(nx, mx): return sqrt(float(factorial(nx - abs(mx)))/factorial(nx + abs(mx)))
    
    cosv = np.zeros(3 * llimit)
    sinv = np.zeros(3 * llimit)
    for mx in range(-llimit, llimit+1):
        cosv[mx] = cos(-1.0 * mx * sph[2])
        sinv[mx] = sin(-1.0 * mx * sph[2])

    for lx in range(L):
        scipy_p = lpmv(range(lx+1), lx, cos(sph[1]))
        radn = 1.0 / (sph[0] ** (lx+1))
        for mx in range(-lx, lx+1):
            coeff = charge * radn * Hfoo(lx, mx) * scipy_p[abs(mx)] 
            arr[re_lm(lx, mx)] += cosv[mx] * coeff
            arr[im_lm(lx, mx)] += sinv[mx] * coeff












