#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

import os

# everyone likes global vars
cwd = os.getcwd()

record_ppmd = os.path.join(cwd, 'record_ppmd')
record_lmps = os.path.join(cwd, 'record_lmps')
record_dlpoly = os.path.join(cwd, 'record_dlpoly')

ppmd = np.loadtxt(record_ppmd)
lmps = np.loadtxt(record_lmps)
dlpoly = np.loadtxt(record_dlpoly)


p_ppmd = plt.loglog(ppmd[::,0], ppmd[::,1], 'ro--', label='Framework')
p_lmps = plt.loglog(lmps[::,0], lmps[::,1], 'bs--', label='Lammps')
p_dlpoly = plt.loglog(dlpoly[::,0], dlpoly[::,1], 'g^--', label='DLPOLY')

plt.legend()

fh = open(os.path.join(os.getcwd(), 'config_dir/SIMULATION_RECORD'), 'r')
_sim_rec = fh.read()
fh.close()

_sim_rec = _sim_rec.split()
_sim = 'N = %(N)s, Steps = %(I)s, Reduced density = %(R)s, Cutoff = %(C)s' % {'N': _sim_rec[0], 
                                                                              'I': _sim_rec[1], 
                                                                              'R': _sim_rec[2],
                                                                              'C': _sim_rec[3]}


plt.title("Strong scaling comparison (log-log)\n" + _sim)
plt.xlabel("Number of cores")
plt.ylabel("Total time taken for integration (s)")


labels = [int(l) for l in ppmd[::,0]]
plt.xticks(ppmd[::,0], labels)

plt.show()








