"""
Methods for Coulombic forces and energies with the classical Ewald method.
"""

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

from math import sqrt, log, ceil, pi, exp, cos, sin, erfc, floor
import numpy as np
import ctypes
import ppmd.kernel
import ppmd.loop
import ppmd.runtime
import ppmd.data
import ppmd.access

from ppmd.cuda import CUDA_IMPORT

if CUDA_IMPORT:
    import ppmd.cuda
    import ppmd.cuda.cuda_base as cubase
    import ppmd.cuda.cuda_data as cudata
    from ppmd.cuda.cuda_loop import ParticleLoop as cuParticleLoop

import cmath
import scipy
import scipy.special
from scipy.constants import epsilon_0
import time

from ppmd import host

_charge_coulomb = scipy.constants.physical_constants['atomic unit of charge'][0]


class EwaldOrthoganalCuda(object):

    def __init__(self, domain, eps=10.**-6, real_cutoff=None, alpha=None, recip_cutoff=None, recip_nmax=None, shared_memory=False):

        self.domain = domain
        self.eps = float(eps)


        ss = cmath.sqrt(scipy.special.lambertw(1./eps)).real

        if alpha is None and real_cutoff is None:
            real_cutoff = 10.
            alpha = (ss/real_cutoff)**2.
        elif alpha is not None and real_cutoff is not None:
            ss = real_cutoff * sqrt(alpha)

        elif alpha is None:
            alpha = (ss/real_cutoff)**2.
        else:
            self.real_cutoff = ss/sqrt(alpha)

        self.real_cutoff = float(real_cutoff)
        """Real space cutoff"""
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

        # CUDA parts
        cuda_block_size = 16
        self._subvars['SUB_BLOCKSIZE'] = str(cuda_block_size)
        self._vars['cuda_block_size'] = cuda_block_size
        self._vars['cuda_part_tmp'] = cudata.ParticleDat(ncomp=2, dtype=ctypes.c_double)
        self._vars['cuda_block_tmp'] = cubase.Array(ncomp=cuda_block_size*2, dtype=ctypes.c_double)

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

        self._cont_lib = ppmd.loop.ParticleLoop(
            kernel=_cont_kernel,
            dat_dict={
                'Positions': ppmd.data.PlaceHolderDat(ncomp=3, dtype=ctypes.c_double)(
                    ppmd.access.READ),
                'Charges': ppmd.data.PlaceHolderDat(ncomp=1, dtype=ctypes.c_double)(
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

        self._extract_force_energy_lib = ppmd.loop.ParticleLoop(
            kernel=_cont_kernel,
            dat_dict={
                'Positions': ppmd.data.PlaceHolderDat(ncomp=3, dtype=ctypes.c_double)(
                    ppmd.access.READ),
                'Forces': ppmd.data.PlaceHolderDat(ncomp=3, dtype=ctypes.c_double)(
                    ppmd.access.WRITE),
                'Energy': ppmd.data.PlaceHolderArray(ncomp=1, dtype=ctypes.c_double)(
                    ppmd.access.INC_ZERO),
                'Charges': ppmd.data.PlaceHolderDat(ncomp=1, dtype=ctypes.c_double)(
                    ppmd.access.READ),
                'RecipSpace': self._vars['recip_space_kernel'](
                    ppmd.access.READ),
                'CoeffSpace': self._vars['coeff_space_kernel'](
                    ppmd.access.READ)
            }
        )

        # cuda basis block
        with open(str(
                ppmd.runtime.LIB_DIR) + '/CudaEwaldOrthSource/BasisBlock.h', 'r') as fh:
            _cont_header_src = fh.read()
        _cont_header = (ppmd.kernel.Header(block=_cont_header_src % self._subvars),)

        with open(str(
                ppmd.runtime.LIB_DIR) + '/CudaEwaldOrthSource/BasisBlock.cu', 'r') as fh:
            _cont_source = fh.read()

        _cont_kernel = ppmd.kernel.Kernel(
            name='CUDA_BasisBlock',
            code=_cont_source,
            headers=_cont_header,
            static_args={
                'kxmin': ctypes.c_int,
                'kxmax': ctypes.c_int,
                'kymin': ctypes.c_int,
                'kzmin': ctypes.c_int,
                'kx': ctypes.c_int,
                'ky': ctypes.c_int,
                'kz': ctypes.c_int

            }
        )

        self._cu_basis_lib = cuParticleLoop(
            kernel=_cont_kernel,
            dat_dict={
                'Positions': cudata.PlaceHolderDat(ncomp=3, dtype=ctypes.c_double)(
                    ppmd.access.READ),
                'Charges': cudata.PlaceHolderDat(ncomp=1, dtype=ctypes.c_double)(
                    ppmd.access.READ),
                'BasisBlock': self._vars['cuda_part_tmp'](ppmd.access.WRITE),
                'RecipBlock': self._vars['cuda_block_tmp'](ppmd.access.INC)
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

        self._real_space_pairloop = ppmd.pairloop.PairLoopNeighbourListNS(
            kernel=_real_kernel,
            dat_dict={
                'P': ppmd.data.PlaceHolderDat(ncomp=3, dtype=ctypes.c_double)(ppmd.access.READ),
                'Q': ppmd.data.PlaceHolderDat(ncomp=1, dtype=ctypes.c_double)(ppmd.access.READ),
                'F': ppmd.data.PlaceHolderDat(ncomp=3, dtype=ctypes.c_double)(ppmd.access.INC),
                'u': ppmd.data.PlaceHolderArray(ncomp=1, dtype=ctypes.c_double)(ppmd.access.INC)
            },
            shell_cutoff=1.05*self.real_cutoff
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


    def _calculate_reciprocal_contribution(self, positions, charges):

        NLOCAL = positions.npart_local

        nkmax = self._vars['nmax_vec']
        blocksize = self._vars['cuda_block_size']

        if self._vars['cuda_part_tmp'].npart < NLOCAL:
            self._vars['cuda_part_tmp'].resize(NLOCAL)

        xmax = int(floor(nkmax[0]/blocksize)+1)

        for iz in range(-1*nkmax[2],nkmax[2]+1):
            for iy in range(-1*nkmax[1],nkmax[1]+1):
                for ixx in range(xmax):
                    ix = -1*nkmax[0] + ixx*blocksize
                    self._cu_basis_lib.execute(
                        n = NLOCAL,
                        dat_dict={
                            'Positions': positions(ppmd.access.READ),
                            'Charges': charges(ppmd.access.READ),
                            'BasisBlock': self._vars['cuda_part_tmp'](ppmd.access.WRITE),
                            'RecipBlock': self._vars['cuda_block_tmp'](ppmd.access.INC)
                        },
                        static_args={
                            'kxmin': ctypes.c_int(-1*nkmax[0]),
                            'kxmax': ctypes.c_int(nkmax[0]),
                            'kymin': ctypes.c_int(-1*nkmax[1]),
                            'kzmin': ctypes.c_int(-1*nkmax[2]),
                            'kx': ctypes.c_int(ix),
                            'ky': ctypes.c_int(iy),
                            'kz': ctypes.c_int(iz)

                        }
                    )



    def _extract_reciprocal_contribution(self, positions, charges, forces, energy):

        NLOCAL = positions.npart_local

        self._extract_force_energy_lib.execute(
            n = NLOCAL,
            dat_dict={
                'Positions': positions(ppmd.access.READ),
                'Charges': charges(ppmd.access.READ),
                'RecipSpace': self._vars['recip_space_kernel'](
                    ppmd.access.READ),
                'Forces': forces(ppmd.access.WRITE),
                'Energy': energy(ppmd.access.INC_ZERO),
                'CoeffSpace': self._vars['coeff_space_kernel'](
                    ppmd.access.READ)
            }
        )

    def extract_forces_energy_reciprocal(self, positions, charges, forces, energy):
        self._extract_reciprocal_contribution(positions, charges, forces, energy)



    def evaluate_contributions(self, positions, charges):

        t0 = time.time()

        self._calculate_reciprocal_contribution(positions, charges)


    def extract_forces_energy_real(self, positions, charges, forces, energy):

        if self._real_space_pairloop is None:
            self._init_real_space_lib()

        self._real_space_pairloop.execute(
            dat_dict={
                'P': positions(ppmd.access.READ),
                'Q': charges(ppmd.access.READ),
                'F': forces(ppmd.access.INC),
                'u': energy(ppmd.access.INC)
            }
        )

    def _init_self_interaction_lib(self):
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

        self._self_interaction_lib = ppmd.loop.ParticleLoop(
            kernel=_real_kernel,
            dat_dict={
                'Q': ppmd.data.PlaceHolderDat(ncomp=1, dtype=ctypes.c_double)(ppmd.access.READ),
                'u': ppmd.data.PlaceHolderArray(ncomp=1, dtype=ctypes.c_double)(ppmd.access.INC)
            }
        )

    def evaluate_self_interactions(self, charges, energy):

        if self._self_interaction_lib is None:
            self._init_self_interaction_lib()
        self._self_interaction_lib.execute(
            dat_dict={
                'Q': charges(ppmd.access.READ),
                'u': energy(ppmd.access.INC)
            }
        )


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


