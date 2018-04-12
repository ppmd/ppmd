
import pytest
from ppmd import *
import numpy as np

np.set_printoptions(threshold=np.nan)


def test_simple_shift():
    n = 4
    N = n**3
    e = (3., 4., 5.)
    la = utility.lattice.cubic_lattice((n,n,n), e)
    
    S = state.State()
    S.npart = N
    S.domain = domain.BaseDomainHalo(extent=e)
    S.domain.boundary_condition = domain.BoundaryTypePeriodic()

    S.P  = data.PositionDat(ncomp=3)
    S.P1 = data.ParticleDat(ncomp=3)


    la_shift = la + 26.9999999999999999

    la_wrap = utility.sanitise.wrap_positions(e, la_shift)
    
    for px in range(N):
        pt = la[px, :]
        assert -0.5 * e[0] <= pt[0]
        assert  0.5 * e[0] >= pt[0]
        assert -0.5 * e[1] <= pt[1]
        assert  0.5 * e[1] >= pt[1]
        assert -0.5 * e[2] <= pt[2]
        assert  0.5 * e[2] >= pt[2]       
        pt = la_wrap[px, :]
        assert -0.5 * e[0] <= pt[0]
        assert  0.5 * e[0] >= pt[0]
        assert -0.5 * e[1] <= pt[1]
        assert  0.5 * e[1] >= pt[1]
        assert -0.5 * e[2] <= pt[2]
        assert  0.5 * e[2] >= pt[2]

    S.P[:N:, :] = la_wrap
    S.P1[:N:, :] = la
    
    # if la_wrap contains particles outside the domain this call will error
    # with a lost particle error
    S.filter_on_domain_boundary()


