"""
Methods for Coulombic forces and energies.
"""

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

import cmath
import scipy
import scipy.interpolate
import scipy.special
from scipy.constants import epsilon_0
import time
import ewald

from ppmd import host

#import matplotlib.pyplot as plt

_charge_coulomb = scipy.constants.physical_constants['atomic unit of charge'][0]

class EwaldOrthoganalFFT(ewald.EwaldOrthoganal):


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

        for rz in xrange(nmax_vec[2]+1):
            for ry in xrange(nmax_vec[1]+1):
                for rx in xrange(nmax_vec[0]+1):
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

        recip_space = self._vars['recip_space_kernel']
        self._cont_lib.execute(
            n = NLOCAL,
            dat_dict={
                'Positions': positions(ppmd.access.READ),
                'Charges': charges(ppmd.access.READ),
                'RecipSpace': recip_space(ppmd.access.INC_ZERO)
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
        eps = 10.**-6
        alpha = self.alpha

        N = positions.npart_local
        E = self.domain.extent
        Eo2x = E[0]*0.5
        Eo2y = E[1]*0.5
        Eo2z = E[2]*0.5

        nkx = self.kmax[0]*2 + 2
        nky = self.kmax[1]*2 + 2
        nkz = self.kmax[2]*2 + 2

        nkx = 40
        nky = 40
        nkz = 40

        hx = E[0]/nkx
        hy = E[1]/nky
        hz = E[2]/nkz

        self.fftrealspace = np.zeros((nkz, nky, nkx))

        rmax = sqrt((-1.0/alpha)*log(((alpha/pi)**(-3./2.))* eps ))
        ssx = int(rmax/hx)
        ssy = int(rmax/hy)
        ssz = int(rmax/hz)

        print "NK", nkx, nky, nkz, ssx, ssy, ssz

        for px in xrange(N):

            qi = charges[px, 0]

            rx = positions[px, 0] + Eo2x
            ry = positions[px, 1] + Eo2y
            rz = positions[px, 2] + Eo2z

            cx = int(rx/hx)
            cy = int(ry/hy)
            cz = int(rz/hz)

            for oz in xrange(-1*ssz, ssz+1):
                ozv = (cz+oz)*hz
                for oy in xrange(-1*ssy, ssy+1):
                    oyv = (cy+oy)*hy
                    for ox in xrange(-1*ssx, ssx+1):
                        oxv = (cx+ox)*hx
                        x = rx - oxv
                        y = ry - oyv
                        z = rz - ozv

                        val = qi * (alpha/pi)**(3./2)*exp(-1*alpha*(x*x+y*y+z*z))

                        #print "zyx", (cz+oz)%nkz,(cy+oy)%nky ,(cx+ox)%nkx, "ozyx", oz, oy, ox
                        self.fftrealspace[
                            (cz+oz)%nkz,(cy+oy)%nky ,(cx+ox)%nkx
                        ] += val

            #print "R", rz, ry, rx, "C", cz, cy, cx




        #print self.fftrealspace

        print self.recip_vectors

        freqx = nkx * np.fft.fftfreq(nkx) * 2*pi/E[0]
        freqy = nky * np.fft.fftfreq(nky) * 2*pi/E[1]
        freqz = nkz * np.fft.fftfreq(nkz) * 2*pi/E[2]

        print freqx

        self.fftkspace = np.fft.fftn(self.fftrealspace)

        kx, ky, kz = np.meshgrid(freqx, freqy, freqz)

        k2 = kx**2 + ky**2 + kz**2
        k2[0,0,0] = 1.0
        k2 = 4.*pi/k2

        self.fftr2space = np.fft.ifftn(k2*self.fftkspace)

        X = np.linspace(-0.5*hx,E[0]+0.5*hx, nkx+2)
        Y = np.linspace(-0.5*hy,E[1]+0.5*hy, nky+2)
        Z = np.linspace(-0.5*hz,E[2]+0.5*hz, nkz+2)

        # middle
        interp_space = np.zeros((nkz+2, nky+2, nkx+2))
        interp_space[1:-1:, 1:-1:, 1:-1:] = self.fftr2space[:,:,:].real

        # corners
        interp_space[-1, -1, -1] = self.fftr2space[0,0,0].real
        interp_space[ 0, -1, -1] = self.fftr2space[-1,0,0].real
        interp_space[-1,  0, -1] = self.fftr2space[0,-1,0].real
        interp_space[ 0,  0, -1] = self.fftr2space[-1,-1,0].real
        interp_space[-1, -1,  0] = self.fftr2space[0,0,-1].real
        interp_space[ 0, -1,  0] = self.fftr2space[-1,0,-1].real
        interp_space[-1,  0,  0] = self.fftr2space[0,-1,-1].real
        interp_space[ 0,  0,  0] = self.fftr2space[-1,-1,-1].real

        # planes
        interp_space[0, 1:-1:, 1:-1:] = self.fftr2space[-1, :,:].real
        interp_space[-1, 1:-1:, 1:-1:] = self.fftr2space[0, :,:].real

        interp_space[1:-1:, 0, 1:-1:] = self.fftr2space[:, -1,:].real
        interp_space[1:-1:, -1, 1:-1:] = self.fftr2space[:, 0,:].real

        interp_space[1:-1:, 1:-1:, 0] = self.fftr2space[:, :, -1].real
        interp_space[1:-1:, 1:-1:, -1] = self.fftr2space[:, :, 0].real

        # edges
        interp_space[-1, -1, 1:-1:] = self.fftr2space[ 0, 0, :].real
        interp_space[ 0, -1, 1:-1:] = self.fftr2space[-1, 0, :].real
        interp_space[ 0,  0, 1:-1:] = self.fftr2space[-1,-1, :].real
        interp_space[-1,  0, 1:-1:] = self.fftr2space[ 0,-1, :].real

        # edges
        interp_space[-1, 1:-1:, -1] = self.fftr2space[ 0,:,  0,].real
        interp_space[ 0, 1:-1:, -1] = self.fftr2space[-1,:,  0,].real
        interp_space[ 0, 1:-1:,  0] = self.fftr2space[-1,:, -1,].real
        interp_space[-1, 1:-1:,  0] = self.fftr2space[ 0,:, -1,].real

        # edges
        interp_space[1:-1:, -1, -1] = self.fftr2space[:, 0,  0,].real
        interp_space[1:-1:,  0, -1] = self.fftr2space[:,-1,  0,].real
        interp_space[1:-1:,  0,  0] = self.fftr2space[:,-1, -1,].real
        interp_space[1:-1:, -1,  0] = self.fftr2space[:, 0, -1,].real



        I = scipy.interpolate.RegularGridInterpolator(
            (Z,Y,X),
            interp_space,
            bounds_error = False,
            fill_value=None,
            method='linear'
        )


        ZYX = np.zeros((N,3))
        ZYX[:, 0] = positions[:, 2] + Eo2x
        ZYX[:, 1] = positions[:, 1] + Eo2y
        ZYX[:, 2] = positions[:, 0] + Eo2z


        u = np.dot(
            charges[:,0],
            I(ZYX)
        )

        print u

        u1 = 0.0
        u2 = 0.0
        for px in xrange(N):
            qi = charges[px, 0]

            rx = positions[px, 0] + Eo2x
            ry = positions[px, 1] + Eo2y
            rz = positions[px, 2] + Eo2z

            cx = int(rx/hx)
            cy = int(ry/hy)
            cz = int(rz/hz)

            val1 = I((rz,ry,rx))*qi
            val2 = self.fftr2space[cz, cy, cx].real*qi

            u1 += val1
            u2 += val2

        print u1, u2

        print u1*self.internal_to_ev()*0.5, 0.5*u2*self.internal_to_ev(), u*self.internal_to_ev()*0.5







        '''
        print ((-2./(hx*hx)) + (-2./(hy*hy)) + (-2./(hz*hz))), E[0], hx, nkx
        diff = ((2./(hx*hx)) + (2./(hy*hy)) + (2./(hz*hz)))*self.fftr2space[:,:,:].real -\
        (1./(hx*hx)) * (np.roll(self.fftr2space.real, 1, axis=2) + np.roll(self.fftr2space.real, -1, axis=2)) -\
        (1./(hy*hy)) * (np.roll(self.fftr2space.real, 1, axis=1) + np.roll(self.fftr2space.real, -1, axis=1)) -\
        (1./(hz*hz)) * (np.roll(self.fftr2space.real, 1, axis=0) + np.roll(self.fftr2space.real, -1, axis=0))
        pl2 = plt.matshow(diff[0, :,:])
        plt.colorbar(pl2)
        plt.figure(1)
        pl1 = plt.matshow(self.fftr2space[0, :,:].real)
        plt.colorbar(pl1)
        plt.figure(1)
        pl0 = plt.matshow(self.fftrealspace[0, :,:])
        plt.colorbar(pl0)
        plt.show()
        print "norm", np.linalg.norm(diff - self.fftrealspace)/ (nkx*nky*nkz)
        '''





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

    ######################################################
    # Below here is python based test code
    ######################################################




