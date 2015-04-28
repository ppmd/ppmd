#!/usr/bin/python

import domain
import potential
import state
import numpy as np
import math
import method
import data
import time
import constant
import kernel
import ctypes


if __name__ == '__main__':
    ctypes_map = {ctypes.c_double:'double'}
    a = data.ScalarArray()
    print a.dtype
    print ctypes_map[a.dtype]
