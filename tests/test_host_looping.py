#!/usr/bin/python
"""
single a multi process bcs
"""



import pytest
import ctypes
import numpy as np
import ppmd as md

N = 1000
E = 8.
Eo2 = E/2.

@pytest.fixture
def state():
    A = State()
    A.npart = N
    A.domain = md.domain.BaseDomainHalo(extent=(E,E,E))
    A.p = PositionDat(ncomp=3)
    A.v = ParticleDat(ncomp=3)
    A.f = ParticleDat(ncomp=3)
    A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    A.u = ScalarArray(ncomp=2)
    A.u.halo_aware = True

    return A

@pytest.fixture
def s_nd():
    """
    State with no domain, hence will not spatially decompose
    """

    A = State()
    A.npart = N
    A.p = PositionDat(ncomp=3)
    A.v = ParticleDat(ncomp=3)
    A.f = ParticleDat(ncomp=3)
    A.gid = ParticleDat(ncomp=1, dtype=ctypes.c_int)

    A.u = ScalarArray(ncomp=2)
    A.u.halo_aware = True

    return A

'''
1) Put particles on edge.
2) Test BCs nothing should move
3) Apply loop to move outside
4) reapply
'''

def test_host_bc1(state):
    
    # need to put particles dx/2 from boundary




    pi = np.random.uniform(-1*Eo2, Eo2, [N,3])
    vi = np.random.normal(0, 2, [N,3])
    fi = np.zeros([N,3])
    gidi = np.arange(N)

    s_nd.p[:] = pi
    s_nd.v[:] = vi
    s_nd.f[:] = fi
    s_nd.gid[:,0] = gidi














