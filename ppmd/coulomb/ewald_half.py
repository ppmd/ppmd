"""
Methods for Coulombic forces and energies with the classical Ewald method.
"""
from __future__ import division, print_function
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import ctypes, os
import ppmd.kernel
import ppmd.loop
import ppmd.runtime
import ppmd.data
import ppmd.access

from ewald import EwaldOrthoganal

_SRC_DIR = os.path.dirname(os.path.realpath(__file__))

class EwaldOrthoganalHalf(EwaldOrthoganal):

    def _init_libs(self):

        # reciprocal contribution calculation
        with open(str(
                _SRC_DIR) + '/EwaldOrthSource/AccumulateRecipHalf.h', 'r') as fh:
            _cont_header_src = fh.read()
        _cont_header = ppmd.kernel.Header(block=_cont_header_src % self._subvars)

        with open(str(
                _SRC_DIR) + '/EwaldOrthSource/AccumulateRecipHalf.cpp', 'r') as fh:
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
                _SRC_DIR) + '/EwaldOrthSource/ExtractForceEnergyHalf.h', 'r') as fh:
            _cont_header_src = fh.read()
        _cont_header = ppmd.kernel.Header(block=_cont_header_src % self._subvars)

        with open(str(
                _SRC_DIR) + '/EwaldOrthSource/ExtractForceEnergyHalf.cpp', 'r') as fh:
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


