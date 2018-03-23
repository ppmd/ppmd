from __future__ import print_function, division, absolute_import
"""
Methods for Coulombic forces and energies with the classical Ewald method.
"""
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import ctypes, os

from ppmd import kernel, loop, data, access, pairloop
from ppmd.coulomb.ewald import EwaldOrthoganal

_SRC_DIR = os.path.dirname(os.path.realpath(__file__))

class EwaldOrthoganalHalf(EwaldOrthoganal):

    def _init_libs(self):

        # reciprocal contribution calculation
        with open(str(
                _SRC_DIR) + '/EwaldOrthSource/AccumulateRecipHalf.h', 'r') as fh:
            _cont_header_src = fh.read()
        _cont_header = kernel.Header(block=_cont_header_src % self._subvars)

        with open(str(
                _SRC_DIR) + '/EwaldOrthSource/AccumulateRecipHalf.cpp', 'r') as fh:
            _cont_source = fh.read()

        _cont_kernel = kernel.Kernel(
            name='reciprocal_contributions',
            code=_cont_source,
            headers=_cont_header
        )

        if self.shared_memory in ('thread', 'omp'):
            PL = loop.ParticleLoopOMP
        else:
            PL = loop.ParticleLoop

        self._cont_lib = PL(
            kernel=_cont_kernel,
            dat_dict={
                'Positions': data.ParticleDat(ncomp=3, dtype=ctypes.c_double)(
                    access.READ),
                'Charges': data.ParticleDat(ncomp=1, dtype=ctypes.c_double)(
                    access.READ),
                'RecipSpace': self._vars['recip_space_kernel'](
                    access.INC_ZERO)
            }
        )

        # reciprocal extract forces plus energy
        with open(str(
                _SRC_DIR) + '/EwaldOrthSource/ExtractForceEnergyHalf.h', 'r') as fh:
            _cont_header_src = fh.read()
        _cont_header = kernel.Header(block=_cont_header_src % self._subvars)

        with open(str(
                _SRC_DIR) + '/EwaldOrthSource/ExtractForceEnergyHalf.cpp', 'r') as fh:
            _cont_source = fh.read()

        _cont_kernel = kernel.Kernel(
            name='reciprocal_force_energy',
            code=_cont_source,
            headers=_cont_header
        )

        self._extract_force_energy_lib = PL(
            kernel=_cont_kernel,
            dat_dict={
                'Positions': data.ParticleDat(ncomp=3, dtype=ctypes.c_double)(access.READ),
                'Forces': data.ParticleDat(ncomp=3, dtype=ctypes.c_double)(access.INC),
                'Energy': self._vars['recip_space_energy'](access.INC_ZERO),
                'Charges': data.ParticleDat(ncomp=1, dtype=ctypes.c_double)(access.READ),
                'RecipSpace': self._vars['recip_space_kernel'](access.READ),
                'CoeffSpace': self._vars['coeff_space_kernel'](access.READ)
            }
        )

        # reciprocal extract forces plus energy plus per particle energy
        with open(str(
                _SRC_DIR) + '/EwaldOrthSource/ExtractForceEnergyHalfPot.h', 'r') as fh:
            _cont_header_src = fh.read()
        _cont_header = kernel.Header(block=_cont_header_src % self._subvars)

        with open(str(
                _SRC_DIR) + '/EwaldOrthSource/ExtractForceEnergyHalfPot.cpp', 'r') as fh:
            _cont_source = fh.read()

        _cont_kernel = kernel.Kernel(
            name='reciprocal_force_energy',
            code=_cont_source,
            headers=_cont_header
        )

        self._extract_force_energy_pot_lib = PL(
            kernel=_cont_kernel,
            dat_dict={
                'Positions': data.ParticleDat(ncomp=3, dtype=ctypes.c_double)(access.READ),
                'Forces': data.ParticleDat(ncomp=3, dtype=ctypes.c_double)(access.INC),
                'Energy': self._vars['recip_space_energy'](access.INC_ZERO),
                'Charges': data.ParticleDat(ncomp=1, dtype=ctypes.c_double)(access.READ),
                'Potential': data.ParticleDat(ncomp=1, dtype=ctypes.c_double)(access.INC),
                'RecipSpace': self._vars['recip_space_kernel'](access.READ),
                'CoeffSpace': self._vars['coeff_space_kernel'](access.READ)
            }
        )

        # reciprocal potential field
        with open(str(
                _SRC_DIR) + '/EwaldOrthSource/EvaluateFarPotentialField.h', 'r') as fh:
            _cont_header_src = fh.read()
        _cont_header = kernel.Header(block=_cont_header_src % self._subvars)

        with open(str(
                _SRC_DIR) + '/EwaldOrthSource/EvaluateFarPotentialField.cpp', 'r') as fh:
            _cont_source = fh.read()

        _cont_kernel = kernel.Kernel(
            name='reciprocal_potential_field',
            code=_cont_source,
            headers=_cont_header
        )

        self._far_potential_field_lib = PL(
            kernel=_cont_kernel,
            dat_dict={
                'Positions': data.ParticleDat(ncomp=3, dtype=ctypes.c_double)(access.READ),
                'Energy': data.ParticleDat(ncomp=1, dtype=ctypes.c_double)(access.INC_ZERO),
                'mask': data.ParticleDat(ncomp=1, dtype=ctypes.c_int)(access.READ),
                'RecipSpace': self._vars['recip_space_kernel'](access.READ),
                'CoeffSpace': self._vars['coeff_space_kernel'](access.READ)
            }
        )

    def _init_near_potential_lib(self):

        # real space energy and force kernel
        with open(str(
                _SRC_DIR) + '/EwaldOrthSource/EvaluateNearPotentialField.h',
                'r') as fh:
            _cont_header_src = fh.read()
        _cont_header = (kernel.Header(block=_cont_header_src % self._subvars),)

        with open(str(
                _SRC_DIR) + '/EwaldOrthSource/EvaluateNearPotentialField.cpp',
                'r') as fh:
            _cont_source = fh.read()

        _real_kernel = kernel.Kernel(
            name='real_space_part',
            code=_cont_source,
            headers=_cont_header
        )

        if self.shell_width is None:
            rn = self.real_cutoff*1.05
        else:
            rn = self.real_cutoff + self.shell_width

        PPL = pairloop.CellByCellOMP

        self._near_potential_field = PPL(
            kernel=_real_kernel,
            dat_dict={
                'P': data.ParticleDat(ncomp=3, dtype=ctypes.c_double)(access.READ),
                'Q': data.ParticleDat(ncomp=1, dtype=ctypes.c_double)(access.READ),
                'M': data.ParticleDat(ncomp=1, dtype=ctypes.c_int)(access.READ),
                'u': data.ParticleDat(ncomp=1, dtype=ctypes.c_double)(access.INC),
            },
            shell_cutoff=rn
        )


    def evaluate_potential_field(self, charges, points, values, masks):

        if self._real_space_pairloop is None:
            self._init_real_space_lib()

        print("0", values[0])
        NLOCAL = points.npart_local
        re = self._vars['recip_space_energy']
        self._far_potential_field_lib.execute(
            n = NLOCAL,
            dat_dict={
                'mask': masks(access.READ),
                'Positions': points(access.READ),
                'RecipSpace': self._vars['recip_space_kernel'](access.READ),
                'Energy': values(access.INC_ZERO),
                'CoeffSpace': self._vars['coeff_space_kernel'](access.READ)
            }
        )

        print("1", values[0])

        self._init_near_potential_lib()
        self._near_potential_field.execute(
            dat_dict={
                'P': points(access.READ),
                'Q': charges(access.READ),
                'M': masks(access.READ),
                'u': values(access.INC)
            }
        )

        print("2", values[0])












