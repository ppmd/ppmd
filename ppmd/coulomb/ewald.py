"""
Methods for Coulombic forces and energies with the classical Ewald method.
"""
from __future__ import division, print_function
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

from math import sqrt, log, ceil, pi, exp, cos, sin, erfc
import numpy as np
import ctypes
import ppmd.kernel
import ppmd.loop
import ppmd.runtime
import ppmd.data
import ppmd.access
import ppmd.opt as opt

import cmath
import scipy
import scipy.special
from scipy.constants import epsilon_0
import time

from ppmd import host

_charge_coulomb = scipy.constants.physical_constants['atomic unit of charge'][0]


def compute_alpha(N, L, ratio):
    """
    Compute an optimal alpha to give O(N^{3/2}) scaling for ewald summation. 
    tau_r: Real space time cost per pair of particles (time cost per real space
    kernel executin)
    tau_f: Time for reciprocal component per particle per reciprocal vector.
    :param N: Number of particles
    :param L: Domain extent (assumes cubic)
    :param ratio: ratio: tau_r/tau_f. 
    :return: optimal alpha
    """
    return ((float(ratio)) * (pi**3.) * float(N) / (float(L)**6.))**(1./3.)


def compute_rc_nc_cutoff(alpha, L, eps=10.**-6):
    ss = cmath.sqrt(scipy.special.lambertw(1./eps)).real
    rc = ss/sqrt(alpha)
    nc = int(round(ss*L*sqrt(alpha)/pi))
    return rc, nc



class EwaldOrthoganal(object):

    def __init__(self, domain, eps=10.**-6, real_cutoff=None, alpha=None, recip_cutoff=None, recip_nmax=None, shared_memory=False, shell_width=None, work_ratio=1.0):
        # type: (object, object, object, object, object, object, object, object, object) -> object

        self.domain = domain
        self.eps = float(eps)


        assert shared_memory in (False, 'omp', 'mpi')

        ss = cmath.sqrt(scipy.special.lambertw(1./eps)).real

        if alpha is not None and real_cutoff is not None:
            ss = real_cutoff * sqrt(alpha)

        elif alpha is None:
            alpha = (ss/real_cutoff)**2.
        else:
            real_cutoff = ss/sqrt(alpha)


        assert alpha is not None, "no alpha deduced/passed"
        assert real_cutoff is not None, "no real_cutoff deduced/passed"


        self.real_cutoff = float(real_cutoff)
        """Real space cutoff"""
        self.shell_width = shell_width
        """Real space padding width"""
        self.alpha = float(alpha)
        """alpha"""


        # these parts are specific to the orthongonal box
        extent = self.domain.extent
        lx = (extent[0], 0., 0.)
        ly = (0., extent[1], 0.)
        lz = (0., 0., extent[2])
        ivolume = 1./np.dot(lx, np.cross(ly, lz))


        gx = np.cross(ly,lz)*ivolume * 2. * pi
        gy = np.cross(lz,lx)*ivolume * 2. * pi
        gz = np.cross(lx,ly)*ivolume * 2. * pi

        sqrtalpha = sqrt(alpha)

        nmax_x = round(ss*extent[0]*sqrtalpha/pi)
        nmax_y = round(ss*extent[1]*sqrtalpha/pi)
        nmax_z = round(ss*extent[2]*sqrtalpha/pi)


        #print gx, gy, gz
        #print 'nmax:', nmax_x, nmax_y, nmax_z
        #print "alpha", alpha, "sqrt(alpha)", sqrtalpha


        gxl = np.linalg.norm(gx)
        gyl = np.linalg.norm(gy)
        gzl = np.linalg.norm(gz)
        if recip_cutoff is None:
            max_len = min(
                gxl*float(nmax_x),
                gyl*float(nmax_y),
                gzl*float(nmax_z)
            )
        else:
            max_len = recip_cutoff

        if recip_nmax is None:
            nmax_x = int(ceil(max_len/gxl))
            nmax_y = int(ceil(max_len/gyl))
            nmax_z = int(ceil(max_len/gzl))
        else:
            nmax_x = recip_nmax[0]
            nmax_y = recip_nmax[1]
            nmax_z = recip_nmax[2]


        #print 'max reciprocal vector len:', max_len
        nmax_t = max(nmax_x, nmax_y, nmax_z)
        #print "nmax_t", nmax_t

        self.kmax = (nmax_x, nmax_y, nmax_z)
        """Number of reciporcal vectors taken in each direction."""
        self.recip_cutoff = max_len
        """Reciprocal space cutoff."""
        self.recip_vectors = (gx,gy,gz)
        """Reciprocal lattice vectors"""
        self.ivolume = ivolume

        opt.PROFILE[self.__class__.__name__+':recip_vectors'] = (self.recip_vectors)
        opt.PROFILE[self.__class__.__name__+':recip_cutoff'] = (self.recip_cutoff)
        opt.PROFILE[self.__class__.__name__+':recip_kmax'] = (self.kmax)
        opt.PROFILE[self.__class__.__name__+':alpha'] = (self.alpha)
        opt.PROFILE[self.__class__.__name__+':tol'] = (eps)
        opt.PROFILE[self.__class__.__name__+':real_cutoff'] = (self.real_cutoff)

        # define persistent vars
        self._vars = {}
        self._vars['alpha']           = ctypes.c_double(alpha)
        self._vars['max_recip']       = ctypes.c_double(max_len)
        self._vars['nmax_vec']        = host.Array((nmax_x, nmax_y, nmax_z), dtype=ctypes.c_int)
        self._vars['recip_vec']       = host.Array(np.zeros((3,3), dtype=ctypes.c_double))
        self._vars['recip_vec'][0, :] = gx
        self._vars['recip_vec'][1, :] = gy
        self._vars['recip_vec'][2, :] = gz
        self._vars['ivolume'] = ivolume
        self._vars['coeff_space_kernel'] = ppmd.data.ScalarArray(
            ncomp=((nmax_x+1)*(nmax_y+1)*(nmax_z+1)),
            dtype=ctypes.c_double
        )
        self._vars['coeff_space'] = self._vars['coeff_space_kernel'].data.view().reshape(nmax_z+1, nmax_y+1, nmax_x+1)
        #self._vars['coeff_space'] = np.zeros((nmax_z+1, nmax_y+1, nmax_x+1), dtype=ctypes.c_double)

        # pass stride in tmp space vector
        self._vars['recip_axis_len'] = ctypes.c_int(nmax_t)

        # |axis | planes | quads
        reciplen = (nmax_t+1)*12 +\
                   8*nmax_x*nmax_y + \
                   8*nmax_y*nmax_z +\
                   8*nmax_z*nmax_x +\
                   16*nmax_x*nmax_y*nmax_z

        self._vars['recip_space_kernel'] = ppmd.data.GlobalArray(
            size=reciplen,
            dtype=ctypes.c_double,
            shared_memory=shared_memory
        )
        self._vars['recip_space_energy'] = ppmd.data.GlobalArray(
            size=1,
            dtype=ctypes.c_double,
            shared_memory=shared_memory
        )
        self._vars['real_space_energy'] = ppmd.data.GlobalArray(
            size=1,
            dtype=ctypes.c_double,
            shared_memory=shared_memory
        )
        self._vars['self_interaction_energy'] = ppmd.data.GlobalArray(
            size=1,
            dtype=ctypes.c_double,
            shared_memory=shared_memory
        )


        self.shared_memory = shared_memory

        #self._vars['recip_vec_kernel'] = data.ScalarArray(np.zeros(3, dtype=ctypes.c_double))
        #self._vars['recip_vec_kernel'][0] = gx[0]
        #self._vars['recip_vec_kernel'][1] = gy[1]
        #self._vars['recip_vec_kernel'][2] = gz[2]

        self._subvars = dict()
        self._subvars['SUB_GX'] = str(gx[0])
        self._subvars['SUB_GY'] = str(gy[1])
        self._subvars['SUB_GZ'] = str(gz[2])
        self._subvars['SUB_NKMAX'] = str(nmax_t)
        self._subvars['SUB_NK'] = str(nmax_x)
        self._subvars['SUB_NL'] = str(nmax_y)
        self._subvars['SUB_NM'] = str(nmax_z)
        self._subvars['SUB_NKAXIS'] =str(nmax_t)
        self._subvars['SUB_LEN_QUAD'] = str(nmax_x*nmax_y*nmax_z)
        self._subvars['SUB_MAX_RECIP'] = str(max_len)
        self._subvars['SUB_MAX_RECIP_SQ'] = str(max_len**2.)
        self._subvars['SUB_SQRT_ALPHA'] = str(sqrt(alpha))
        self._subvars['SUB_REAL_CUTOFF_SQ'] = str(real_cutoff**2.)
        self._subvars['SUB_REAL_CUTOFF'] = str(real_cutoff)
        self._subvars['SUB_M_SQRT_ALPHA_O_PI'] = str(-1.0*sqrt(alpha/pi))
        self._subvars['SUB_M2_SQRT_ALPHAOPI'] = str(-2.0*sqrt(alpha/pi))
        self._subvars['SUB_MALPHA'] = str(-1.0*alpha)

        self._real_space_pairloop = None
        self._init_libs()
        self._init_coeff_space()
        self._self_interaction_lib = None

    def _init_libs(self):

        # reciprocal contribution calculation
        with open(str(
                ppmd.runtime.LIB_DIR) + '/EwaldOrthSource/AccumulateRecip.h', 'r') as fh:
            _cont_header_src = fh.read()
        _cont_header = ppmd.kernel.Header(block=_cont_header_src % self._subvars)

        with open(str(
                ppmd.runtime.LIB_DIR) + '/EwaldOrthSource/AccumulateRecip.cpp', 'r') as fh:
            _cont_source = fh.read()

        _cont_kernel = ppmd.kernel.Kernel(
            name='reciprocal_contributions',
            code=_cont_source,
            headers=_cont_header
        )

        if self.shared_memory in ('thread', 'omp'):
            PL = ppmd.loop.ParticleLoopOMP
        else:
            PL = ppmd.loop.ParticleLoop

        self._cont_lib = PL(
            kernel=_cont_kernel,
            dat_dict={
                'Positions': ppmd.data.ParticleDat(ncomp=3, dtype=ctypes.c_double)(
                    ppmd.access.READ),
                'Charges': ppmd.data.ParticleDat(ncomp=1, dtype=ctypes.c_double)(
                    ppmd.access.READ),
                'RecipSpace': self._vars['recip_space_kernel'](
                    ppmd.access.INC_ZERO)
            }
        )


        # reciprocal extract forces plus energy
        with open(str(
                ppmd.runtime.LIB_DIR) + '/EwaldOrthSource/ExtractForceEnergy.h', 'r') as fh:
            _cont_header_src = fh.read()
        _cont_header = ppmd.kernel.Header(block=_cont_header_src % self._subvars)

        with open(str(
                ppmd.runtime.LIB_DIR) + '/EwaldOrthSource/ExtractForceEnergy.cpp', 'r') as fh:
            _cont_source = fh.read()

        _cont_kernel = ppmd.kernel.Kernel(
            name='reciprocal_force_energy',
            code=_cont_source,
            headers=_cont_header
        )


        self._extract_force_energy_lib = PL(
            kernel=_cont_kernel,
            dat_dict={
                'Positions': ppmd.data.ParticleDat(ncomp=3, dtype=ctypes.c_double)(ppmd.access.READ),
                'Forces': ppmd.data.ParticleDat(ncomp=3, dtype=ctypes.c_double)(ppmd.access.INC),
                'Energy': self._vars['recip_space_energy'](ppmd.access.INC_ZERO),
                'Charges': ppmd.data.ParticleDat(ncomp=1, dtype=ctypes.c_double)(ppmd.access.READ),
                'RecipSpace': self._vars['recip_space_kernel'](ppmd.access.READ),
                'CoeffSpace': self._vars['coeff_space_kernel'](ppmd.access.READ)
            }
        )


    def _init_real_space_lib(self):

        # real space energy and force kernel
        with open(str(
                ppmd.runtime.LIB_DIR) + '/EwaldOrthSource/RealSpaceForceEnergy.h', 'r') as fh:
            _cont_header_src = fh.read()
        _cont_header = (ppmd.kernel.Header(block=_cont_header_src % self._subvars),)

        with open(str(
                ppmd.runtime.LIB_DIR) + '/EwaldOrthSource/RealSpaceForceEnergy.cpp', 'r') as fh:
            _cont_source = fh.read()

        _real_kernel = ppmd.kernel.Kernel(
            name='real_space_part',
            code=_cont_source,
            headers=_cont_header
        ) 

        if self.shell_width is None:
            rn = self.real_cutoff*1.05
        else:
            rn = self.real_cutoff + self.shell_width

        if self.shared_memory in ('thread', 'omp'):
            PPL = ppmd.pairloop.PairLoopNeighbourListNSOMP
        else:
            PPL = ppmd.pairloop.PairLoopNeighbourListNS

        self._real_space_pairloop = PPL(
            kernel=_real_kernel,
            dat_dict={
                'P': ppmd.data.ParticleDat(ncomp=3, dtype=ctypes.c_double)(ppmd.access.READ),
                'Q': ppmd.data.ParticleDat(ncomp=1, dtype=ctypes.c_double)(ppmd.access.READ),
                'F': ppmd.data.ParticleDat(ncomp=3, dtype=ctypes.c_double)(ppmd.access.INC),
                'u': self._vars['real_space_energy'](ppmd.access.INC_ZERO)
            },
            shell_cutoff=rn
        )


    def _init_coeff_space(self):
        recip_vec = self._vars['recip_vec']
        nmax_vec = self._vars['nmax_vec']

        coeff_space = self._vars['coeff_space']
        max_recip = self._vars['max_recip'].value
        alpha = self._vars['alpha'].value
        ivolume = self._vars['ivolume']

        max_recip2 = max_recip**2.
        base_coeff1 = 4.*pi*ivolume
        base_coeff2 = -1./(4.*alpha)


        coeff_space[0,0,0] = 0.0

        count = 0
        for rz in range(nmax_vec[2]+1):
            for ry in range(nmax_vec[1]+1):
                for rx in range(nmax_vec[0]+1):
                    if not (rx == 0 and ry == 0 and rz == 0):

                        rlen2 = (rx*recip_vec[0,0])**2. + \
                                (ry*recip_vec[1,1])**2. + \
                                (rz*recip_vec[2,2])**2.

                        if rlen2 > max_recip2:
                            coeff_space[rz,ry,rx] = 0.0
                        else:
                            count += 8
                            coeff_space[rz,ry,rx] = (base_coeff1/rlen2)*exp(rlen2*base_coeff2)

        opt.PROFILE[self.__class__.__name__+':recip_vector_count'] = \
            (count)


    def _calculate_reciprocal_contribution(self, positions, charges):

        NLOCAL = positions.npart_local

        recip_space = self._vars['recip_space_kernel']
        self._cont_lib.execute(
            n = NLOCAL,
            dat_dict={
                'Positions': positions(ppmd.access.READ),
                'Charges': charges(ppmd.access.READ),
                'RecipSpace': recip_space(ppmd.access.INC_ZERO)
            }
        )

    def _extract_reciprocal_contribution(self, positions, charges, forces, energy=None):

        NLOCAL = positions.npart_local
        re = self._vars['recip_space_energy']
        self._extract_force_energy_lib.execute(
            n = NLOCAL,
            dat_dict={
                'Positions': positions(ppmd.access.READ),
                'Charges': charges(ppmd.access.READ),
                'RecipSpace': self._vars['recip_space_kernel'](ppmd.access.READ),
                'Forces': forces(ppmd.access.INC),
                'Energy': re(ppmd.access.INC_ZERO),
                'CoeffSpace': self._vars['coeff_space_kernel'](ppmd.access.READ)
            }
        )
        if energy is not None:
            energy[0] = re[0]
        return re[0]

    def extract_forces_energy_reciprocal(self, positions, charges, forces, energy):
        self._extract_reciprocal_contribution(positions, charges, forces, energy)


    def evaluate_contributions(self, positions, charges):

        t0 = time.time()

        self._calculate_reciprocal_contribution(positions, charges)


    def extract_forces_energy_real(self, positions, charges, forces, energy=None):

        if self._real_space_pairloop is None:
            self._init_real_space_lib()

        re = self._vars['real_space_energy']
        self._real_space_pairloop.execute(
            dat_dict={
                'P': positions(ppmd.access.READ),
                'Q': charges(ppmd.access.READ),
                'F': forces(ppmd.access.INC),
                'u': re(ppmd.access.INC_ZERO)
            }
        )

        if energy is not None:
            energy[0] = re[0]
        return re[0]



    def _init_self_interaction_lib(self):

        if self.shared_memory in ('thread', 'omp'):
            PL = ppmd.loop.ParticleLoopOMP
        else:
            PL = ppmd.loop.ParticleLoop

        with open(str(
            ppmd.runtime.LIB_DIR) + '/EwaldOrthSource/SelfInteraction.h', 'r') as fh:
            _cont_header_src = fh.read()
        _cont_header = (ppmd.kernel.Header(block=_cont_header_src % self._subvars),)

        with open(str(
            ppmd.runtime.LIB_DIR) + '/EwaldOrthSource/SelfInteraction.cpp', 'r') as fh:
            _cont_source = fh.read()

        _real_kernel = ppmd.kernel.Kernel(
            name='real_space_part',
            code=_cont_source,
            headers=_cont_header
        )

        self._self_interaction_lib = PL(
            kernel=_real_kernel,
            dat_dict={
                'Q': ppmd.data.ParticleDat(ncomp=1, dtype=ctypes.c_double)(ppmd.access.READ),
                'u': self._vars['self_interaction_energy'](ppmd.access.INC_ZERO)
            }
        )

    def evaluate_self_interactions(self, charges, energy=None):

        if self._self_interaction_lib is None:
            self._init_self_interaction_lib()
        en = self._vars['self_interaction_energy']
        self._self_interaction_lib.execute(
            dat_dict={
                'Q': charges(ppmd.access.READ),
                'u': en(ppmd.access.INC_ZERO)
            }
        )
        if energy is not None:
            energy[0] = en[0]

        return en[0]


    @staticmethod
    def internal_to_ev():
        """
        Multiply by this constant to convert from internal units to eV.
        """
        epsilon_0 = scipy.constants.epsilon_0
        pi = scipy.constants.pi
        c0 = scipy.constants.physical_constants['atomic unit of charge'][0]
        l0 = 10.**-10
        return c0 / (4.*pi*epsilon_0*l0)

    ######################################################
    # Below here is python based test code
    ######################################################

    def _test_python_structure_factor(self, positions=None, charges=None):

        recip_space = self._vars['recip_space_kernel']


        # evaluate coefficient space ------------------------------------------
        nmax_x = self._vars['nmax_vec'][0]
        nmax_y = self._vars['nmax_vec'][1]
        nmax_z = self._vars['nmax_vec'][2]
        recip_axis_len = self._vars['recip_axis_len'].value
        self._vars['recip_axis'] = np.zeros((2,2*recip_axis_len+1,3), dtype=ctypes.c_double)
        self._vars['recip_space'] = np.zeros((2, 2*nmax_x+1, 2*nmax_y+1, 2*nmax_z+1), dtype=ctypes.c_double)

        coeff_space = self._vars['coeff_space']


        # ---------------------------------------------------------------------
        nkmax = self._vars['recip_axis_len'].value
        nkaxis = nkmax


        axes_size = 12*nkaxis
        axes = recip_space[0:axes_size:].view()

        plane_size = 4*nmax_x*nmax_y + 4*nmax_y*nmax_z + 4*nmax_z*nmax_x
        planes = recip_space[axes_size:axes_size+plane_size*2:].view()

        quad_size = nmax_x*nmax_y*nmax_z
        quad_start = axes_size+plane_size*2
        quads = recip_space[quad_start:quad_start+quad_size*16].view()

        # compute energy from structure factor
        engs = 0.


        # AXES ------------------------

        #+ve X
        rax = 0
        iax = 6
        rtmp = axes[rax*nkaxis:(rax+1)*nkaxis:]**2.
        itmp = axes[iax*nkaxis:(iax+1)*nkaxis:]**2.
        engs += np.dot(coeff_space[0,0,1:nmax_x+1:], rtmp[:nmax_x:] + itmp[:nmax_x:])
        #for ix in range(nmax_x):
        #    engs += coeff_space[0,0,ix+1]*(rtmp[ix] + itmp[ix])

        # -ve X
        rax = 2
        iax = 8
        rtmp = axes[rax*nkaxis:(rax+1)*nkaxis:]**2.
        itmp = axes[iax*nkaxis:(iax+1)*nkaxis:]**2.
        engs += np.dot(coeff_space[0,0,1:nmax_x+1:], rtmp[:nmax_x:] + itmp[:nmax_x:])
        #for ix in range(nmax_x):
        #    engs += coeff_space[0,0,ix+1]*(rtmp[ix] + itmp[ix])


        #+ve y
        rax = 1
        iax = 7
        rtmp = axes[rax*nkaxis:(rax+1)*nkaxis:]**2.
        itmp = axes[iax*nkaxis:(iax+1)*nkaxis:]**2.
        engs += np.dot(coeff_space[0,1:nmax_y+1:,0], rtmp[:nmax_y:] + itmp[:nmax_y:])
        #for iy in range(nmax_y):
        #    engs += coeff_space[0,iy+1,0]*(rtmp[iy] + itmp[iy])


        # -ve y
        rax = 3
        iax = 9
        rtmp = axes[rax*nkaxis:(rax+1)*nkaxis:]**2.
        itmp = axes[iax*nkaxis:(iax+1)*nkaxis:]**2.
        engs += np.dot(coeff_space[0,1:nmax_y+1:,0], rtmp[:nmax_y:] + itmp[:nmax_y:])
        #for iy in range(nmax_y):
        #    engs += coeff_space[0,iy+1,0]*(rtmp[iy] + itmp[iy])


        #+ve z
        rax = 4
        iax = 10
        rtmp = axes[rax*nkaxis:(rax+1)*nkaxis:]**2.
        itmp = axes[iax*nkaxis:(iax+1)*nkaxis:]**2.
        engs += np.dot(coeff_space[1:nmax_z+1:,0,0], (rtmp[:nmax_z:] + itmp[:nmax_z:]))
        #for iz in range(nmax_z):
        #    engs += coeff_space[iz+1,0,0]*(rtmp[iz] + itmp[iz])



        # -ve z
        rax = 5
        iax = 11
        rtmp = axes[rax*nkaxis:(rax+1)*nkaxis:]**2.
        itmp = axes[iax*nkaxis:(iax+1)*nkaxis:]**2.
        engs += np.dot(coeff_space[1:nmax_z+1:,0,0], (rtmp[:nmax_z:] + itmp[:nmax_z:]))
        #for iz in range(nmax_z):
        #    engs += coeff_space[iz+1,0,0]*(rtmp[iz] + itmp[iz])



        # PLANES -----------------------



        # XY
        tps = nmax_x*nmax_y*4
        rplane = 0
        iplane = tps
        rtmp = planes[rplane:rplane+tps:]**2.
        itmp = planes[iplane:iplane+tps:]**2.

        for px in range(4):
            tmp_rtmp = rtmp[px::4]
            tmp_itmp = itmp[px::4]
            for iy in range(nmax_y):
                for ix in range(nmax_x):
                    tmpc = coeff_space[0, iy+1, ix+1]
                    engs += tmpc*(tmp_rtmp[iy*nmax_x+ix] + tmp_itmp[iy*nmax_x+ix])


        # YZ
        rplane = iplane + tps
        tps = nmax_y*nmax_z*4
        iplane = rplane + tps
        rtmp = planes[rplane:rplane+tps:]**2.
        itmp = planes[iplane:iplane+tps:]**2.

        for px in range(4):
            tmp_rtmp = rtmp[px::4]
            tmp_itmp = itmp[px::4]
            for iz in range(nmax_z):
                for iy in range(nmax_y):
                    tmpc = coeff_space[iz+1, iy+1, 0]
                    engs += tmpc*(tmp_rtmp[iz*nmax_y+iy] + tmp_itmp[iz*nmax_y+iy])


        # ZX
        rplane = iplane + tps
        tps = nmax_z*nmax_x*4
        iplane = rplane + tps
        rtmp = planes[rplane:rplane+tps:]**2.
        itmp = planes[iplane:iplane+tps:]**2.

        for px in range(4):
            tmp_rtmp = rtmp[px::4]
            tmp_itmp = itmp[px::4]
            for ix in range(nmax_x):
                for iz in range(nmax_z):
                    tmpc = coeff_space[iz+1, 0, ix+1]
                    engs += tmpc*(tmp_rtmp[ix*nmax_z+iz] + tmp_itmp[ix*nmax_z+iz])


        # guadrants
        rquads = quads[:8*quad_size:]**2.
        iquads = quads[8*quad_size::]**2.

        for qx in range(8):
            tmp_rquad = rquads[qx::8]
            tmp_iquad = iquads[qx::8]
            for iz in range(nmax_z):
                for iy in range(nmax_y):
                    for ix in range(nmax_x):
                        tmpc = coeff_space[iz+1, iy+1, ix+1]
                        engs+=tmpc*(tmp_rquad[nmax_x*(nmax_y*iz + iy) + ix] +
                                    tmp_iquad[nmax_x*(nmax_y*iz + iy) + ix])

        t1 = time.time()
        #print "time taken total", t1-t0, "time taken reciprocal", self._cont_lib.loop_timer.time
        return engs*0.5


    @staticmethod
    def _COMP_EXP_PACKED(x, gh):
        gh[0] = cos(x)
        gh[1] = sin(x)

    @staticmethod
    def _COMP_AB_PACKED(a,x,gh):
        gh[0] = a[0]*x[0] - a[1]*x[1]
        gh[1] = a[0]*x[1] + a[1]*x[0]

    @staticmethod
    def _COMP_ABC_PACKED(a,x,k,gh):

        axmby = a[0]*x[0] - a[1]*x[1]
        xbpay = a[0]*x[1] + a[1]*x[0]

        gh[0] = axmby*k[0] - xbpay*k[1]
        gh[1] = axmby*k[1] + xbpay*k[0]


    def test_evaluate_python_lr(self, positions, charges):
        # python version for sanity
        # recpirocal space PYTHON TODO remove when C working
        nmax_x = self._vars['nmax_vec'][0]
        nmax_y = self._vars['nmax_vec'][1]
        nmax_z = self._vars['nmax_vec'][2]
        # tmp space vector

        recip_axis_len = self._vars['recip_axis_len'].value
        self._vars['recip_axis'] = np.zeros((2,2*recip_axis_len+1,3), dtype=ctypes.c_double)
        self._vars['recip_space'] = np.zeros((2, 2*nmax_x+1, 2*nmax_y+1, 2*nmax_z+1), dtype=ctypes.c_double)

        np.set_printoptions(linewidth=400)

        N_LOCAL = positions.npart_local

        recip_axis = self._vars['recip_axis']
        recip_vec = self._vars['recip_vec']
        nmax_vec = self._vars['nmax_vec']
        recip_space = self._vars['recip_space']

        coeff_space = self._vars['coeff_space']
        max_recip = self._vars['max_recip'].value
        max_recip2 = max_recip*max_recip
        alpha = self._vars['alpha'].value
        ivolume = self._vars['ivolume']

        #print "recip_axis_len", recip_axis_len

        recip_space[:] = 0.0

        t0 = time.time()

        gx = recip_vec[0,0]
        gy = recip_vec[1,1]
        gz = recip_vec[2,2]

        #N_LOCAL = 1
        for lx in range(N_LOCAL):

            for dx in range(3):
                ri =  -1.0 * recip_vec[dx,dx] * positions[lx, dx]

                # unit at middle as exp(0) = 1+0i
                recip_axis[:, recip_axis_len, dx] = (1.,0.)


                # first positive index along each axis
                self._COMP_EXP_PACKED(
                    ri,
                    recip_axis[:, recip_axis_len+1, dx]
                )

                # zeroth index on each axis
                recip_axis[0, recip_axis_len-1, dx] = 1.0
                recip_axis[1, recip_axis_len-1, dx] = 0.0

                # first negative index on each axis
                base_el = recip_axis[:, recip_axis_len+1, dx]
                recip_axis[0, recip_axis_len-1, dx] = base_el[0]
                recip_axis[1, recip_axis_len-1, dx] = -1. * base_el[1]

                # check vals
                #cval = cmath.exp(-1. * 1j*ri)
                #_ctol15(recip_axis[0,recip_axis_len-1, dx], cval.real, "recip base -1")
                #_ctol15(recip_axis[1,recip_axis_len-1, dx], cval.imag, "recip base -1")
                #cval = cmath.exp(1j*ri)
                #_ctol15(recip_axis[0,recip_axis_len+1, dx], cval.real, "recip base 1")
                #_ctol15(recip_axis[1,recip_axis_len+1, dx], cval.imag, "recip base 1")


                # +ve part
                for ex in range(2+recip_axis_len, nmax_vec[dx]+recip_axis_len+1):
                    self._COMP_AB_PACKED(
                        base_el,
                        recip_axis[:,ex-1,dx],
                        recip_axis[:,ex,dx]
                    )

                    # check val
                    #cval = cmath.exp(1j*ri)**(ex - recip_axis_len)
                    #_ctol15(recip_axis[0,ex, dx], cval.real, "recip base")
                    #_ctol15(recip_axis[1,ex, dx], cval.imag, "recip base")


                # rest of axis
                for ex in range(recip_axis_len-1):
                    recip_axis[0,recip_axis_len-2-ex,dx] = recip_axis[0,recip_axis_len+2+ex,dx]
                    recip_axis[1,recip_axis_len-2-ex,dx] = -1. * recip_axis[1,recip_axis_len+2+ex,dx]
                    ##check val
                    #cval = cmath.exp(-1. * 1j*ri)**( 2 + ex )
                    #_ctol15(recip_axis[0,recip_axis_len-2-ex, dx], cval.real, "recip base")
                    #_ctol15(recip_axis[1,recip_axis_len-2-ex, dx], cval.imag, "recip base")

            # now calculate the contributions to all of recip space
            qx = charges[lx, 0]
            tmp = np.zeros(2, dtype=ctypes.c_double)

            for rz in range(2*nmax_vec[2]+1):
                rzp = abs(rz-nmax_vec[2])
                recip_len2 = (rzp*gz)**2.
                for ry in range(2*nmax_vec[1]+1):
                    ryp = abs(ry-nmax_vec[1])
                    recip_len2zy = recip_len2 + (ryp*gy)**2.
                    for rx in range(2*nmax_vec[0]+1):
                        rxp = abs(rx-nmax_vec[0])
                        recip_len2zyx = recip_len2zy + (rxp*gx)**2.

                        if (recip_len2zyx <= max_recip2) or rxp == 0 or ryp == 0 or rzp == 0:

                            tmp[:] = 0.0
                            self._COMP_ABC_PACKED(
                                recip_axis[:,rx,0],
                                recip_axis[:,ry,1],
                                recip_axis[:,rz,2],
                                tmp[:]
                            )
                            recip_space[:,rx,ry,rz] += tmp[:]*qx
                        else:

                            recip_space[:,rx,ry,rz] = 0.0



        t1 = time.time()


        # evaluate coefficient space ------------------------------------------
        max_recip2 = max_recip**2.
        base_coeff1 = 4.*pi*ivolume
        base_coeff2 = -1./(4.*alpha)

        coeff_space[0,0,0] = 0.0
        for rz in range(nmax_vec[2]+1):
            for ry in range(nmax_vec[1]+1):
                for rx in range(nmax_vec[0]+1):
                    if not (rx == 0 and ry == 0 and rz == 0):

                        rlen2 = (rx*recip_vec[0,0])**2. + \
                                (ry*recip_vec[1,1])**2. + \
                                (rz*recip_vec[2,2])**2.

                        if rlen2 > max_recip2:
                            coeff_space[rz,ry,rx] = 0.0
                        else:
                            coeff_space[rz,ry,rx] = (base_coeff1/rlen2)*exp(rlen2*base_coeff2)

        # ---------------------------------------------------------------------
        # evaluate total long range contribution loop over each particle then
        # over the reciprocal space

        t2 = time.time()


        eng = 0.
        eng_im = 0.

        for px in range(N_LOCAL):

            rx = positions[px, 0]
            ry = positions[px, 1]
            rz = positions[px, 2]
            qx = charges[px,0]

            for kz in range(2*nmax_vec[2]+1):
                rzkz = rz*recip_vec[2,2]*(kz-nmax_vec[2])

                for ky in range(2*nmax_vec[1]+1):
                    ryky = ry*recip_vec[1,1]*(ky-nmax_vec[1])
                    for kx in range(2*nmax_vec[0]+1):
                        rxkx = rx*recip_vec[0,0]*(kx-nmax_vec[0])

                        rxp = abs(kx-nmax_vec[0])
                        ryp = abs(ky-nmax_vec[1])
                        rzp = abs(kz-nmax_vec[2])

                        recip_len2 = (rxp*gx)**2. + (ryp*gy)**2. + (rzp*gz)**2.
                        if (recip_len2 <= max_recip2) or rxp == 0 or ryp == 0 or rzp == 0:

                            coeff = coeff_space[
                                abs(kz-nmax_vec[2]),
                                abs(ky-nmax_vec[1]),
                                abs(kx-nmax_vec[0])
                            ] * qx

                            re_coeff = cos(rzkz+ryky+rxkx)*coeff
                            im_coeff = sin(rzkz+ryky+rxkx)*coeff

                            re_con = recip_space[0,kx,ky,kz]
                            im_con = recip_space[1,kx,ky,kz]

                            eng += re_coeff*re_con - im_coeff*im_con
                            eng_im += re_coeff*im_con + im_coeff*re_con




                        # print re_coeff, im_coeff, re_con, im_con



        t3 = time.time()

        # ---------------------------------------------------------------------
        # compute energy from structure factor
        engs = 0.

        for kz in range(2*nmax_vec[2]+1):
            for ky in range(2*nmax_vec[1]+1):
                for kx in range(2*nmax_vec[0]+1):

                    coeff = coeff_space[
                        abs(kz-nmax_vec[2]),
                        abs(ky-nmax_vec[1]),
                        abs(kx-nmax_vec[0])
                    ]

                    re_con = recip_space[0,kx,ky,kz]
                    im_con = recip_space[1,kx,ky,kz]
                    con = re_con*re_con + im_con*im_con

                    engs += coeff*con

        # ---------------------------------------------------------------------

        t4 = time.time()

        #print t1 - t0, t2 - t1, t3 - t2, t4 - t3

        #np.save('co2_recip_space.npy', recip_space)

        return eng*0.5, engs*0.5

    def test_evaluate_python_self(self, charges):

        alpha = self._vars['alpha'].value
        eng_self = np.sum(np.square(charges[:,0]))
        eng_self *= -1. * sqrt(alpha/pi)

        return eng_self




    def test_evaluate_python_sr(self, positions, charges):
        extent = self.domain.extent
        N_LOCAL = positions.npart_local
        alpha = self._vars['alpha'].value
        cutoff2 = self.real_cutoff**2.
        sqrt_alpha = sqrt(alpha)

        # N^2 way for checking.....

        #print N_LOCAL, cutoff2, epsilon_0, extent, alpha

        eng = 0.0
        count = 0

        mind = 10000.

        for ix in range(N_LOCAL):
            ri = positions[ix,:]
            qi = charges[ix, 0]

            for jx in range(ix+1, N_LOCAL):

                rj = positions[jx,:]

                rij = rj - ri

                if abs(rij[0]) > (extent[0]/2.):
                    rij[0] = extent[0] - abs(rij[0])
                if abs(rij[1]) > (extent[1]/2.):
                    rij[1] = extent[1] - abs(rij[1])
                if abs(rij[2]) > (extent[2]/2.):
                    rij[2] = extent[2] - abs(rij[2])

                r2 = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]

                if r2 < cutoff2:
                    len_rij = sqrt(r2)
                    qj = charges[jx, 0]
                    mind = min(mind, len_rij)
                    eng += (qi*qj*erfc(sqrt_alpha*len_rij)/len_rij)
                    #eng += (qi*qj/len_rij)
                    count += 2

        #print count, mind
        return eng


def _test_split1(extent, eps=10.**-6, alpha=None, real_cutoff=None ):

    #ss = cmath.sqrt(scipy.special.lambertw(1./eps)).real


    if alpha is not None and real_cutoff is not None:
        ss = real_cutoff * sqrt(alpha)

    nmax = round(ss*extent*sqrt(alpha)/pi)

    print(real_cutoff, alpha, nmax)



def _ctol(a, b, m='No message given.', tol=10.**-15):
    err = abs(a-b)
    if  err> tol :
        print(err, m)


