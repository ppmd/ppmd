#!/usr/bin/python

import numpy as np
import ppmd
from ppmd.access import *
from ppmd.data import ParticleDat, ScalarArray, GlobalArray, PositionDat

State = ppmd.state.State
BoundaryTypePeriodic = ppmd.domain.BoundaryTypePeriodic
BaseDomainHalo = ppmd.domain.BaseDomainHalo


def test_non_pos_1():
    
    rng = np.random.RandomState(97531)

    E = 1.0
    A = State()
    A.domain = BaseDomainHalo((E,E,E))
    A.domain.boundary_condition = BoundaryTypePeriodic()
    

    N = 10
    A.npart = 10

    A.P = PositionDat()
    A.V = ParticleDat(ncomp=3)
    A.P[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    A.V[:] = rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3))
    A.scatter_data_from(0)
    
    curr_int_id = A.V._vid_int
    new_vel = rng.uniform(size=(A.npart_local, 3))

    with A.V.modify_view() as m:
        assert m.shape[0] == A.npart_local
        assert m.shape[1] == 3
        m[:] = new_vel

    new_int_id = A.V._vid_int
    assert new_int_id > curr_int_id
    assert np.linalg.norm(A.V[:A.npart_local,:] - new_vel, np.inf) < 10.**-16


    with A.V.modify_view() as m:
        pass
    assert A.V._vid_int > new_int_id
    

    

