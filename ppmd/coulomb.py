"""
Methods for Coulombic forces and energies.
"""

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

from math import sqrt, log, ceil, pi, exp, cos, sin, erfc
import numpy as np
import ctypes
import build
import runtime

import cmath
import scipy
import scipy.special

from ppmd import data

class CoulombicEnergy(object):

    def __init__(self, domain, eps=10.**-6, real_cutoff=10., alpha=None):

        self.domain = domain
        self.eps = float(eps)
        self.real_cutoff = float(real_cutoff)

        ss = cmath.sqrt(scipy.special.lambertw(1000000)).real

        if alpha is None:
            alpha = ss/real_cutoff
        else:
            self.real_cutoff = ss/alpha

        # these parts are specific to the orthongonal box
        extent = self.domain.extent
        lx = (extent[0], 0., 0.)
        ly = (0., extent[1], 0.)
        lz = (0., 0., extent[2])
        ivolume = 1./np.dot(lx, np.cross(ly, lz))
        
        gx = np.cross(ly,lz)*ivolume
        gy = np.cross(lz,lx)*ivolume
        gz = np.cross(lx,ly)*ivolume

        nmax_x = ceil(ss*extent[0]*alpha/pi)
        nmax_y = ceil(ss*extent[1]*alpha/pi)
        nmax_z = ceil(ss*extent[2]*alpha/pi)

        print gx, gy, gz
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

        print "recip vector lengths", gxl, gyl, gzl

        nmax_x = int(ceil(max_len/gxl))
        nmax_y = int(ceil(max_len/gyl))
        nmax_z = int(ceil(max_len/gzl))

        print 'max reciprocal vector len:', max_len
        nmax_t = max(nmax_x, nmax_y, nmax_z)
        print "nmax_t", nmax_t

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
        self._vars['recip_axis'] = np.zeros((2,2*nmax_t+1,3), dtype=ctypes.c_double)
        # recpirocal space
        self._vars['recip_space'] = np.zeros((2, 2*nmax_x+1, 2*nmax_y+1, 2*nmax_z+1), dtype=ctypes.c_double)
        self._vars['coeff_space'] = np.zeros((nmax_x+1, nmax_y+1, nmax_z+1), dtype=ctypes.c_double)

        with open(str(runtime.LIB_DIR) + '/CoulombicEnergyOrthSource.h','r') as fh:
            header = fh.read()

        with open(str(runtime.LIB_DIR) + '/CoulombicEnergyOrthSource.cpp','r') as fh:
            source = fh.read()

        self._lib = build.simple_lib_creator(header, source, 'CoulombicEnergyOrth')

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


    def evaluate_python_lr(self, positions, charges):
        # python version for sanity

        np.set_printoptions(linewidth=400)

        N_LOCAL = positions.npart_local

        recip_axis = self._vars['recip_axis']
        recip_vec = self._vars['recip_vec']
        recip_axis_len = self._vars['recip_axis_len'].value
        nmax_vec = self._vars['nmax_vec']
        recip_space = self._vars['recip_space']

        coeff_space = self._vars['coeff_space']
        max_recip = self._vars['max_recip'].value
        alpha = self._vars['alpha'].value

        print "recip_axis_len", recip_axis_len

        recip_space[:] = 0.0

        for lx in range(N_LOCAL):
            print 60*'-'
            print positions[lx, :]

            for dx in range(3):

                gi = -1.0 * recip_vec[dx,dx]
                ri = positions[lx, dx]*gi

                # unit at middle as exp(0) = 1+0i
                recip_axis[:, recip_axis_len, dx] = (1.,0.)


                # first element along each axis
                self._COMP_EXP_PACKED(
                    ri,
                    recip_axis[:, recip_axis_len+1, dx]
                )


                base_el = recip_axis[:, recip_axis_len+1, dx]
                recip_axis[0, recip_axis_len-1, dx] = base_el[0]
                recip_axis[1, recip_axis_len-1, dx] = -1. * base_el[1]


                # +ve part
                for ex in range(2+recip_axis_len, nmax_vec[dx]+recip_axis_len+1):
                    self._COMP_AB_PACKED(
                        base_el,
                        recip_axis[:,ex-1,dx],
                        recip_axis[:,ex,dx]
                    )

                # rest of axis
                for ex in range(recip_axis_len-1):
                    recip_axis[0,recip_axis_len-2-ex,dx] = recip_axis[0,recip_axis_len+2+ex,dx]
                    recip_axis[1,recip_axis_len-2-ex,dx] = -1. * recip_axis[1,recip_axis_len+2+ex,dx]

                print "\t", ri,2*"\n","re", recip_axis[0,:,dx], "\nim", recip_axis[1,:,dx]

            # now calculate the contributions to all of recip space
            qx = charges[lx, 0]
            tmp = np.zeros(2, dtype=ctypes.c_double)
            for rz in xrange(2*nmax_vec[2]+1):
                for ry in xrange(2*nmax_vec[1]+1):
                    for rx in xrange(2*nmax_vec[0]+1):
                        tmp[:] = 0.0
                        self._COMP_ABC_PACKED(
                            recip_axis[:,rx,0],
                            recip_axis[:,ry,1],
                            recip_axis[:,rz,2],
                            tmp[:]
                        )
                        recip_space[:,rx,ry,rz] += tmp[:]*qx

        print 60*"="
        print "re"
        print recip_space[0,:,:,:]
        print "im"
        print recip_space[1,:,:,:]
        # evaluate coefficient space

        max_recip2 = max_recip**2.
        base_coeff1 = 4.*pi
        base_coeff2 = -1./(4.*alpha)

        for rz in xrange(nmax_vec[2]+1):
            for ry in xrange(nmax_vec[1]+1):
                for rx in xrange(nmax_vec[0]+1):
                    if not (rx == 0 and ry == 0 and rz == 0):

                        rlen2 = (rx*recip_vec[0,0])**2. + (ry*recip_vec[1,1])**2. + (rz*recip_vec[2,2])**2.

                        if rlen2 > max_recip2:
                            coeff_space[rx,ry,rz] = 0.0
                        else:
                            coeff_space[rx,ry,rz] = (base_coeff1/rlen2)*exp(rlen2*base_coeff2)

        print 60*'='

        for rz in range(nmax_vec[2]+1):
            print coeff_space[:,:,rz]


        # evaluate total long range contribution loop over each particle then
        # over the reciprocal space

        eng = 0.
        eng_im = 0.

        for px in range(N_LOCAL):

            rx = positions[px, 0]
            ry = positions[px, 1]
            rz = positions[px, 2]

            for kz in xrange(2*nmax_vec[2]+1):
                rzkz = rz*recip_vec[2,2]*(kz-nmax_vec[2])
                for ky in xrange(2*nmax_vec[1]+1):
                    ryky = ry*recip_vec[1,1]*(ky-nmax_vec[1])
                    for kx in xrange(2*nmax_vec[0]+1):
                        rxkx = rx*recip_vec[0,0]*(kx-nmax_vec[0])

                        coeff = coeff_space[
                            abs(kx-nmax_vec[0]),
                            abs(ky-nmax_vec[1]),
                            abs(kz-nmax_vec[2])
                        ] * charges[px,0]

                        re_coeff = cos(rzkz+ryky+rxkx)*coeff
                        im_coeff = sin(rzkz+ryky+rxkx)*coeff

                        re_con = recip_space[0,kx,ky,kz]
                        im_con = recip_space[1,kx,ky,kz]

                        eng += re_coeff*re_con - im_coeff*im_con
                        eng_im += re_coeff*im_con + im_coeff*re_con

                        print re_coeff, im_coeff, re_con, im_con

        print "ENG", eng
        print "iENG", eng_im

        return eng


    def evaluate_python_self(self, charges):

        alpha = self._vars['alpha'].value
        N_LOCAL = charges.npart_local

        eng_self = 0.0
        for px in xrange(N_LOCAL):
            eng_self += charges[px, 0]**2.

        eng_self *= sqrt(alpha/pi)

        return eng_self


    def evaluate_python_sr(self, positions, charges):
        extent = self.domain.extent
        N_LOCAL = positions.npart_local
        alpha = self._vars['alpha'].value
        cutoff2 = self.real_cutoff**2.
        sqrt_alpha = sqrt(alpha)

        # N^2 way for checking.....
        eng = 0.0
        for ix in xrange(N_LOCAL):
            for jx in xrange(N_LOCAL):
                if ix != jx:
                    ri = positions[ix,:]
                    rj = positions[jx,:]
                    qi = charges[ix, 0]
                    qj = charges[jx, 0]

                    rij = rj - ri

                    if rij[0] > extent[0]/2:
                        rij[0] = extent[0] - rij[0]
                    if rij[1] > extent[1]/2:
                        rij[1] = extent[1] - rij[1]
                    if rij[2] > extent[2]/2:
                        rij[2] = extent[2] - rij[2]

                    r2 = rij[0]**2. * rij[1]**2. * rij[2]**2.
                    if r2 < cutoff2:
                        len_rij = sqrt(r2)
                        eng += qi*qj*erfc(sqrt_alpha*len_rij)/len_rij

        eng *= 0.5
        return eng


    def _check_single_contrib(self, pos, charge):
        """
        Check a single particles contribution
        :param pos: 
        :param charge: 
        :return: False on error, True on no error
        """

        recip_axis = self._vars['recip_axis']
        recip_vec = self._vars['recip_vec']
        recip_axis_len = self._vars['recip_axis_len'].value
        nmax_vec = self._vars['nmax_vec']
        recip_space = self._vars['recip_space']

        coeff_space = self._vars['coeff_space']
        max_recip = self._vars['max_recip'].value
        alpha = self._vars['alpha'].value


        lr_eng = self.evaluate_python_lr(pos, charge)
        self_eng = self.evaluate_python_self(charge)

        posc = pos[0,:]

        for rz in xrange(2*nmax_vec[2]+1):
            for ry in xrange(2*nmax_vec[1]+1):
                for rx in xrange(2*nmax_vec[0]+1):

                    rzp = rz - nmax_vec[2]
                    ryp = ry - nmax_vec[1]
                    rxp = rx - nmax_vec[0]

                    kx = np.array([
                        rxp*recip_vec[0,0], ryp*recip_vec[1,1], rzp*recip_vec[2,2]
                    ])

                    mokr = -1. * np.dot(kx, posc)

                    kstr = str(kx)

                    if not abs(charge[0,] * cos(mokr) - recip_space[0,rx,ry,rz]) < 10.**(-15):
                        print "real error " + str(charge[0,] * cos(mokr)) + " " + str(recip_space[0,rx,ry,rz]) + " " + kstr
                        return False
                    if not abs(charge[0,] * sin(mokr) - recip_space[1,rx,ry,rz]) < 10.**(-15):
                        print "imag error " + str(charge[0,] * sin(mokr)) + " " + str(recip_space[1,rx,ry,rz]) + " " + kstr
                        return False



    def _check_quad_contrib(self, testpos, testq):
        """
        Check a single particles contribution
        :param pos: 
        :param charge: 
        :return: 
        """

        extent = self.domain.extent
        pos = data.ParticleDat(ncomp=3, npart=4)
        charge = data.ParticleDat(ncomp=1, npart=4)

        e0 = extent[0]
        e1 = extent[1]

        pos[0,:] = (0.5*e0, 0.5*e1, 0.)
        pos[1,:] = (-0.5*e0, 0.5*e1, 0.)
        pos[2,:] = (-0.5*e0, -0.5*e1, 0.)
        pos[3,:] = (0.5*e0, -0.5*e1, 0.)

        charge[0,] = 1
        charge[1,] = -1
        charge[2,] = 1
        charge[3,] = -1


        recip_axis = self._vars['recip_axis']
        recip_vec = self._vars['recip_vec']
        recip_axis_len = self._vars['recip_axis_len'].value
        nmax_vec = self._vars['nmax_vec']
        recip_space = self._vars['recip_space']

        coeff_space = self._vars['coeff_space']
        max_recip = self._vars['max_recip'].value
        alpha = self._vars['alpha'].value

        lr_eng = self.evaluate_python_lr(pos, charge)
        self_eng = self.evaluate_python_self(charge)

        # evaluate particle in own field this should:
        # a) recalc match other calc
        # b) match the supposid self contribution

        lr_check_contrib_re = 0.0
        lr_check_contrib_im = 0.0

        rx = testpos[0]
        ry = testpos[1]
        rz = testpos[2]

        for px in range(4):
            rx = pos[px,0]
            ry = pos[px,1]
            rz = pos[px,2]
            testtq = charge[px,]

            for kz in xrange(2*nmax_vec[2]+1):
                rzkz = rz*recip_vec[2,2]*(kz-nmax_vec[2])
                for ky in xrange(2*nmax_vec[1]+1):
                    ryky = ry*recip_vec[1,1]*(ky-nmax_vec[1])
                    for kx in xrange(2*nmax_vec[0]+1):
                        rxkx = rx*recip_vec[0,0]*(kx-nmax_vec[0])

                        k2 =  (recip_vec[0,0]*(kx-nmax_vec[0])) ** 2.
                        k2 += (recip_vec[1,1]*(ky-nmax_vec[1])) ** 2.
                        k2 += (recip_vec[2,2]*(kz-nmax_vec[2])) ** 2.

                        if (k2 < (max_recip ** 2.)) and k2 > 10.**-8:
                            ck = exp(-1. * k2 / (4. * (alpha))) *4. * pi / k2
                            ck *= testtq

                            rk = rxkx + ryky + rzkz
                            ckre = ck * cos(rk)
                            ckim = ck * sin(rk)

                            lr_check_contrib_re += ckre*recip_space[0,kx,ky,kx] - ckim*recip_space[1,kx,ky,kz]
                            lr_check_contrib_im += ckre*recip_space[1,kx,ky,kx] + ckim*recip_space[0,kx,ky,kz]


        print "check2"
        print lr_check_contrib_re
        print lr_check_contrib_im
        print self.evaluate_python_self(charge)











