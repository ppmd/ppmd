

from ppmd import lib
from ctypes import *


tlib = lib.build.simple_lib_creator(r'#define TESTING' + lib.common.OMP_DECOMP_HEADER, '')


def wrapper(N, P, threadid):

    tlib.test_set_num_threads(c_int(P))
    tlib.test_set_thread_num(c_int(threadid))

    start = c_int(-1)
    end = c_int(-1)
    
    tlib.get_thread_decomp(c_int(N), byref(start), byref(end))

    return start.value, end.value



def test_decomp_1():
    s, e = wrapper(10, 1, 0)
    assert s == 0
    assert e == 10
    s, e = wrapper(10, 2, 0)
    assert s == 0
    assert e == 5
    s, e = wrapper(10, 2, 1)
    assert s == 5
    assert e == 10

    s, e = wrapper(7, 2, 0)
    assert s == 0
    assert e == 4
    s, e = wrapper(7, 2, 1)
    assert s == 4
    assert e == 7

    s, e = wrapper(3, 4, 0)
    assert s == 0
    assert e == 1

    s, e = wrapper(3, 4, 1)
    assert s == 1
    assert e == 2

    s, e = wrapper(3, 4, 2)
    assert s == 2
    assert e == 3

    s, e = wrapper(3, 4, 3)
    assert s == 3
    assert e == 3

    s, e = wrapper(13, 7, 6)
    assert s == 12
    assert e == 13

    s, e = wrapper(13, 7, 0)
    assert s == 0
    assert e == 2

    s, e = wrapper(13, 7, 5)
    assert s == 10
    assert e == 12
