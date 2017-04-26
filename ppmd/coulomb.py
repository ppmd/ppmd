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
from scipy.constants import epsilon_0
import time

from ppmd import data

_charge_coulomb = scipy.constants.physical_constants['atomic unit of charge'][0]



class CoulombicEnergy(object):

    def __init__(self, domain, eps=10.**-6, real_cutoff=None, alpha=None):

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

        # find shortest nmax_i * gi
        gxl = np.linalg.norm(gx)
        gyl = np.linalg.norm(gy)
        gzl = np.linalg.norm(gz)
        max_len = min(
            gxl*float(nmax_x),
            gyl*float(nmax_y),
            gzl*float(nmax_z)
        )

        #print "recip vector lengths", gxl, gyl, gzl

        nmax_x = int(ceil(max_len/gxl))
        nmax_y = int(ceil(max_len/gyl))
        nmax_z = int(ceil(max_len/gzl))

        #print 'max reciprocal vector len:', max_len
        nmax_t = max(nmax_x, nmax_y, nmax_z)
        #print "nmax_t", nmax_t

        self.kmax = (nmax_x, nmax_y, nmax_z)
        """Number of reciporcal vectors taken in each direction."""
        self.recip_cutoff = max_len
        """Reciprocal space cutoff."""
        self.recip_vectors = (gx,gy,gz)
        """Reciprocal lattice vectors"""


        # define persistent vars
        self._vars = {}
        self._vars['alpha']           = ctypes.c_double(alpha)
        self._vars['max_recip']       = ctypes.c_double(max_len)
        self._vars['nmax_vec']        = np.array((nmax_x, nmax_y, nmax_z), dtype=ctypes.c_int)
        self._vars['recip_vec']       = np.zeros((3,3), dtype=ctypes.c_double)
        self._vars['recip_vec'][0, :] = gx
        self._vars['recip_vec'][1, :] = gy
        self._vars['recip_vec'][2, :] = gz
        self._vars['ivolume'] = ivolume
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

    def test_evaluate_python_lr(self, positions, charges):
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
        ivolume = self._vars['ivolume']

        #print "recip_axis_len", recip_axis_len

        recip_space[:] = 0.0

        t0 = time.time()

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

                        ##check value by direct computation
                        #ri = positions[lx,:]
                        #px = rx - nmax_vec[0]
                        #py = ry - nmax_vec[1]
                        #pz = rz - nmax_vec[2]
                        #gx = recip_vec[0, 0]*px
                        #gy = recip_vec[1, 1]*py
                        #gz = recip_vec[2, 2]*pz

                        #cval = cmath.exp(-1j * np.dot(np.array((gx,gy,gz)),ri))
                        #_ctol(cval.real,tmp[0],'overall check RE',10.**-13)
                        #_ctol(cval.imag,tmp[1],'overall check IM {} {} {}'.format(px,py,pz),10.**-13)

                        ##test abc
                        #tmp2 = np.zeros(2)
                        #self._COMP_ABC_PACKED(
                        #    recip_axis[:,rx,0],
                        #    recip_axis[:,ry,1],
                        #    recip_axis[:,rz,2],
                        #    tmp2[:]
                        #)
                        #cvalx = recip_axis[:,rx,0][0]+1j*recip_axis[:,rx,0][1]
                        #cvaly = recip_axis[:,ry,1][0]+1j*recip_axis[:,ry,1][1]
                        #cvalz = recip_axis[:,rz,2][0]+1j*recip_axis[:,rz,2][1]
                        #cval = cvalx*cvaly*cvalz
                        #_ctol(cval.real, tmp2[0], 'RE')
                        #_ctol(cval.imag, tmp2[1], 'IM')


        t1 = time.time()


        # evaluate coefficient space ------------------------------------------
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
                            coeff_space[rx,ry,rz] = 0.0
                        else:
                            coeff_space[rx,ry,rz] = (base_coeff1/rlen2)*exp(rlen2*base_coeff2)

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

        for kz in xrange(2*nmax_vec[2]+1):
            for ky in xrange(2*nmax_vec[1]+1):
                for kx in xrange(2*nmax_vec[0]+1):

                    coeff = coeff_space[
                        abs(kx-nmax_vec[0]),
                        abs(ky-nmax_vec[1]),
                        abs(kz-nmax_vec[2])
                    ]

                    re_con = recip_space[0,kx,ky,kz]
                    im_con = recip_space[1,kx,ky,kz]
                    con = re_con*re_con + im_con*im_con

                    engs += coeff*con

        # ---------------------------------------------------------------------

        t4 = time.time()

        #print t1 - t0, t2 - t1, t3 - t2, t4 - t3

        return eng*0.5, engs*0.5

    def test_evaluate_python_self(self, charges):

        alpha = self._vars['alpha'].value
        eng_self = np.sum(np.square(charges[:,0]))
        eng_self *= -1. * sqrt(alpha/pi)

        return eng_self

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

        for ix in xrange(N_LOCAL):
            ri = positions[ix,:]
            qi = charges[ix, 0]

            for jx in xrange(ix+1, N_LOCAL):

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

    print real_cutoff, alpha, nmax



def _ctol(a, b, m='No message given.', tol=10.**-15):
    err = abs(a-b)
    if  err> tol :
        print(err, m)


