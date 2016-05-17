#!/usr/bin/python
import hpc_tools

import numpy as np
import matplotlib.pyplot as plt

import os
import math


# everyone likes global vars
cwd = os.getcwd()

record_ppmd = os.path.join(cwd, 'record_ppmd')
record_lmps = os.path.join(cwd, 'record_lmps')
record_dlpoly = os.path.join(cwd, 'record_dlpoly')

ppmd_raw = np.loadtxt(record_ppmd)
lmps_raw = np.loadtxt(record_lmps)
dlpoly_raw = np.loadtxt(record_dlpoly)

ppmd = np.empty_like(ppmd_raw)
lmps = np.empty_like(lmps_raw)
dlpoly = np.empty_like(dlpoly_raw)

for tx in range(ppmd.shape[0]):
    ppmd[tx, 1] = 100. * hpc_tools.scaling.strong_efficiency(ppmd_raw[0,0], ppmd_raw[0,1], ppmd_raw[tx,0], ppmd_raw[tx,1])
    ppmd[tx, 0] = ppmd_raw[tx,0]

for tx in range(lmps.shape[0]):
    lmps[tx, 1] = 100. * hpc_tools.scaling.strong_efficiency(lmps_raw[0,0], lmps_raw[0,1], lmps_raw[tx,0], lmps_raw[tx,1])
    lmps[tx, 0] = lmps_raw[tx,0]

for tx in range(dlpoly.shape[0]):
    dlpoly[tx, 1] = 100. * hpc_tools.scaling.strong_efficiency(dlpoly_raw[0,0], dlpoly_raw[0,1], dlpoly_raw[tx,0], dlpoly_raw[tx,1])

    dlpoly[tx, 0] = dlpoly_raw[tx,0]
print "ppmd"
print ppmd
print "lmps"
print lmps
print "dlpoly"
print dlpoly

fig, ax = plt.subplots()


p_ppmd = plt.semilogx(ppmd[::,0], ppmd[::,1], 'ro--', label='Framework')
p_lmps = plt.semilogx(lmps[::,0], lmps[::,1], 'bs--', label='Lammps')
p_dlpoly = plt.semilogx(dlpoly[::,0], dlpoly[::,1], 'g^--', label='DLPOLY')

plt.legend()

fh = open(os.path.join(os.getcwd(), 'config_dir/SIMULATION_RECORD'), 'r')
_sim_rec = fh.read()
fh.close()

_sim_rec = _sim_rec.split()
_sim = 'N = %(N)s, Steps = %(I)s, Reduced density = %(R)s, Cutoff = %(C)s' % {'N': _sim_rec[0], 
                                                                              'I': _sim_rec[1], 
                                                                              'R': _sim_rec[2],
                                                                              'C': _sim_rec[3]}


plt.title("Parallel Efficiency of Strong Scaling \n" + _sim)
plt.xlabel("\nNumber of cores (top)\nNumber of particle pairs per core (bottom)")
plt.ylabel("Percentage Efficiency (%)")


labels = [int(l) for l in ppmd[::,0]]

npp = (2./3.) * math.pi * ((float(_sim_rec[3]) * 1.1) ** 3.0) * float(_sim_rec[2])

for lx, l in enumerate(labels):
    nn = (float(_sim_rec[0]) / float(l)) * npp



    labels[lx] = str(l) + "\n" + " %1.e " % nn
ax.yaxis.grid(True, which='major')


plt.xticks(ppmd[::,0], labels)

plt.tight_layout()
plt.show()








