"""
Methods for Coulombic forces and energies.
"""

from math import sqrt, log, ceil, pi, exp
import numpy as np
import ctypes
import  build
import runtime

class CoulombicEnergy(object):

    def __init__(self, domain, eps=10.**-6, real_cutoff=10.):
        self.domain = domain
        self.eps = float(eps)
        self.real_cutoff = float(real_cutoff)

        tau = sqrt(abs(eps * self.real_cutoff))
        alpha = sqrt(abs(log(eps*real_cutoff*tau)))*(1.0/real_cutoff)
        tau1 = sqrt(-1.0 * log(eps*self.real_cutoff*(2.*tau*alpha)**2))
        

        # these parts are specific to the orthongonal box
        extent = self.domain.extent
        lx = (extent[0], 0., 0.)
        ly = (0., extent[1], 0.)
        lz = (0., 0., extent[2])
        ivolume = 1./np.dot(lx, np.cross(ly, lz))
        
        gx = np.cross(ly,lz)*ivolume
        gy = np.cross(lz,lx)*ivolume
        gz = np.cross(lx,ly)*ivolume

        nmax_x = ceil(0.25 + np.linalg.norm(gx, ord=2)*alpha*tau1/pi)
        nmax_y = ceil(0.25 + np.linalg.norm(gy, ord=2)*alpha*tau1/pi)
        nmax_z = ceil(0.25 + np.linalg.norm(gz, ord=2)*alpha*tau1/pi)
        

        print 'These nmax values seem too low'
        print eps, tau, tau1, alpha
        print 0.25 + np.linalg.norm(gx, ord=2)*alpha*tau1/pi
        print 0.25 + np.linalg.norm(gx, ord=2)*alpha*tau/pi
        print gx, gy, gz
        print 'nmax:', nmax_x, nmax_y, nmax_z


        print 'Taking tau1 to give nmax until fixed....'
        nmax_x = int(ceil(tau1))
        nmax_y = int(ceil(tau1))
        nmax_z = int(ceil(tau1))

        print 'nmax:', nmax_x, nmax_y, nmax_z
        
        # find shortest nmax_i * gi
        gxl = np.linalg.norm(gx)
        gyl = np.linalg.norm(gy)
        gzl = np.linalg.norm(gz)
        max_len = min(
            gxl*float(nmax_x),
            gyl*float(nmax_y),
            gzl*float(nmax_z)
        )
        
        nmax_x = int(ceil(gxl/max_len))
        nmax_y = int(ceil(gyl/max_len))
        nmax_z = int(ceil(gzl/max_len))

        print 'min reciprocal vector len:', max_len
        nmax_t = max(nmax_x, nmax_y, nmax_z)


        # define persistent vars
        self._vars = {}
        self._vars['alpha']           = ctypes.c_double(alpha)
        self._vars['max_recip']       = ctypes.c_double(max_len)
        self._vars['nmax_vec']        = np.array((nmax_x, nmax_y, nmax_z), dtype=ctypes.c_int)
        self._vars['recip_vec']       = np.zeros((3,3), dtype=ctypes.c_double)
        self._vars['recip_vec'][0, :] = gx
        self._vars['recip_vec'][1, :] = gy
        self._vars['recip_vec'][2, :] = gz
        # Again specific to orthogonal domains
        self._vars['recip_consts'] = np.zeros(3, dtype=ctypes.c_double)
        self._vars['recip_consts'][0] = exp((-1./(4.*alpha)) * (gx[0]**2.) )
        self._vars['recip_consts'][1] = exp((-1./(4.*alpha)) * (gy[1]**2.) )
        self._vars['recip_consts'][2] = exp((-1./(4.*alpha)) * (gz[2]**2.) )
        
        # pass stride in tmp space vector
        self._vars['recip_axis_len'] = ctypes.c_int(nmax_t)
        # tmp space vector
        self._vars['recip_axis'] = np.zeros(6*nmax_t, dtype=ctypes.c_double)
        # recpirocal space
        self._vars['recip_space'] = np.zeros((nmax_x, nmax_y, nmax_z), dtype=ctypes.c_double)



        with open(str(runtime.LIB_DIR) + '/CoulombicEnergyOrthSource.h','r') as fh:
            header = fh.read()

        with open(str(runtime.LIB_DIR) + '/CoulombicEnergyOrthSource.cpp','r') as fh:
            source = fh.read()

        self._lib = build.simple_lib_creator(header, source, 'CoulombicEnergyOrth')


















