#!/usr/bin/python

import os
import ctypes
import numpy as np
import math

from ppmd import *

test_case = 'test_case_0'
file_dir = os.path.join(os.path.curdir, test_case)


# open the field file
fh = open(file_dir + '/FIELD', 'r')
_field = fh.read().split()
fh.close()

# get number of moecules from field file
N = int(_field[_field.index('NUMMOLS')+1])
print 'N:', N

# Get the initial config
fh=open(file_dir + '/CONFIG', 'r')
_config = fh.read()
fh.close()

#get final config
fh=open(file_dir + '/HISTORY', 'r')
_history = fh.read()
fh.close()

# initial and final positions
x0_dl = data.ParticleDat(npart=N, ncomp=3, dtype=ctypes.c_double)
x1_dl = data.ParticleDat(npart=N, ncomp=3, dtype=ctypes.c_double)


#read initial values
_config = _config.splitlines()
_config_positions = [ix.split() for ix in _config[6::4]]

# domain extent
_extent = np.array([_config[2].split()[0], 
                   _config[3].split()[1],
                   _config[4].split()[2]], dtype=ctypes.c_double)

print "Extent --------- \n", _extent


#read final values
_history = _history.splitlines()
_history_positions = [ix.split() for ix in _history[7::2]]

#put values into particle dats
for ix in range(N):
    x0_dl.dat[ix,::] = np.array(_config_positions[ix])
    x1_dl.dat[ix,::] = np.array(_history_positions[ix])

x0_ppmd = fio.xml_to_ParticleDat('test_case_0/ppmd_x0.xml')
x1_ppmd = fio.xml_to_ParticleDat('test_case_0/ppmd_x1.xml')

print "DL_POLY ------------- \n", x0_dl.dat[0:10:,::]
print "PPMD ------------- \n", x0_ppmd.dat[0:10:,::]



# Error between two vectors: L2 norm divided by number of elements.

err0 = 0.0

for ix in range(N):
    diff = np.abs(x0_dl.dat[ix,::] - x0_ppmd.dat[ix,::])
    mask = diff > _extent/2
    diff[mask] = _extent[mask] - diff[mask]
    err0 += np.sum(np.square(diff))

err0 = np.sqrt(err0)
err0 /= N

print "Error 0:", err0, "<-- Test invalid if this is not 0"

err1 = 0.0
max_err1 = 0.0


for ix in range(N):
    diff = np.abs(x1_dl.dat[ix,::] - x1_ppmd.dat[ix,::])
    mask = diff > _extent/2
    diff[mask] = _extent[mask] - diff[mask]
    err1 += np.sum(np.square(diff))
    
    if (np.sum(np.square(diff)) > max_err1 or ix==1652):
        print "New max error", ix, np.sum(np.square(diff)), x1_dl.dat[ix,::] , x1_ppmd.dat[ix,::]



    max_err1 = max(max_err1, np.sum(np.square(diff)))
    


err1 = np.sqrt(err1)
err1 /= N

print "Error 1:", err1
print "Maximum squared error:", max_err1

print "DL_POLY ------------- \n", x1_dl.dat[0:10:,::]
print "PPMD ------------- \n", x1_ppmd.dat[0:10:,::]















