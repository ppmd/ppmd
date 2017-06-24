#!/usr/bin/python

import pytest
import ctypes

from ppmd.access import *
from ppmd.data import ParticleDat, ScalarArray, GlobalArray

P1 = ParticleDat()
P2 = ParticleDat(ncomp=2, dtype=ctypes.c_int)
P3 = ParticleDat(ncomp=1, dtype=ctypes.c_int)
S1 = ScalarArray()
S12 = ScalarArray()
S13 = ScalarArray()
S14 = ScalarArray()
S2 = ScalarArray(ncomp=2, dtype=ctypes.c_int)
G1 = GlobalArray(ncomp=1)
G2 = GlobalArray(ncomp=2, dtype=ctypes.c_int)


allow={
    ParticleDat: (READ, WRITE, INC_ZERO, INC),
    ScalarArray: (READ,),
    GlobalArray: (READ, INC_ZERO, INC)
}

def test_dat_checker_1():
    # check args match allowed args
    with pytest.raises(AssertionError):
        DatArgStore({ScalarArray: (READ,)}, {'A': P1(READ)})
    with pytest.raises(AssertionError):
        DatArgStore({ScalarArray: (READ,)}, {'A': S1(WRITE)})
    DatArgStore({ScalarArray: (READ,)}, {'A': S1(READ)})
    with pytest.raises(AssertionError):
        DatArgStore({ScalarArray: (READ,)}, {1: S1(READ)})
    with pytest.raises(AssertionError):
        DatArgStore({ScalarArray: (READ,)}, {'A1': S1(READ), 'A2': S1(READ)})


def test_dat_checker_2():

    d = DatArgStore({
            ScalarArray: (READ,),
            ParticleDat: (READ, WRITE)
        }, {
            'A': S1(READ),
            'B': P1(WRITE)
        })
    d.items()
    d.items({
        'A': S1(READ),
        'B': P1(WRITE)
    })
    with pytest.raises(AssertionError):
        d.items({
            'A': S1(READ),
            'B': P2(WRITE)
        })
    with pytest.raises(AssertionError):
        d.items({
            'A': S1(READ)
        })
    with pytest.raises(AssertionError):
        d.items({
            'A': S1(READ),
            'B': P3(WRITE)
        })

    with pytest.raises(AssertionError):
        d.items({
            'A': S1(READ),
            'A2': S12(READ),
            'B': P1(WRITE)
        })

    with pytest.raises(AssertionError):
        d.items({
            'A': S1(READ),
            'B': S1(READ)
        })


def test_dat_checker_3():

    d = DatArgStore({
            ScalarArray: (WRITE,),
            ParticleDat: (READ, WRITE),
        }, {
            'B0': S1(WRITE),
            'B1': S12(WRITE),
            'B2': S13(WRITE),
            'B3': S14(WRITE),
        })
    order = d.items()

    order2 = d.items({
        'B2': S13(WRITE),
        'B3': S14(WRITE),
        'B1': S12(WRITE),
        'B0': S1(WRITE),
    })

    for o1, o2 in zip(order, order2):
        assert o1[0] == o2[0], "bad new ordering"
        assert o1[1][0] == o2[1][0], "bad new ordering"
        assert o1[1][1] == o2[1][1], "bad new ordering"


def test_static_checker_1():
    s = StaticArgStore(
        initial={'A': ctypes.c_int}
    )

    with pytest.raises(AssertionError):
        StaticArgStore(
            initial={'A': int}
        )

    assert s.items()[0][0] == 'A'
    assert s.items()[0][1] == ctypes.c_int

    assert type(s.get_args({'A': 4})[0]) is ctypes.c_int, "bad type"







