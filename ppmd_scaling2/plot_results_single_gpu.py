#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import math

import os

# everyone likes global vars
cwd = os.getcwd()

record_ppmd = os.path.join(cwd, 'record_ppmd')
record_lmps = os.path.join(cwd, 'record_lmps')
record_dlpoly = os.path.join(cwd, 'record_dlpoly')


record_ppmd_gpu = os.path.join(cwd, 'record_ppmd_cuda')
record_lmps_gpu = os.path.join(cwd, 'record_lmps_cuda')


ppmd = np.loadtxt(record_ppmd)
lmps = np.loadtxt(record_lmps)
dlpoly = np.loadtxt(record_dlpoly)

ppmd_gpu = np.loadtxt(record_ppmd_gpu)
lmps_gpu = np.loadtxt(record_lmps_gpu)


fig, ax = plt.subplots()

p_ppmd, = plt.loglog(ppmd[::,0], ppmd[::,1], 'ro--', label='Framework')
p_lmps, = plt.loglog(lmps[::,0], lmps[::,1], 'bs--', label='Lammps')
p_dlpoly, = plt.loglog(dlpoly[::,0], dlpoly[::,1], 'g^--', label='DLPOLY')


width = 1.0
p_ppmd_gpu, = plt.bar(ppmd_gpu[0] - width, ppmd_gpu[1], width, color='r', label='Lammps: 1 Node + 1 K20x')
p_lmps_gpu, = plt.bar(lmps_gpu[0], lmps_gpu[1], width, color='b', hatch='x', label='Framework: 1 K20x')


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
plt.xlabel("\nNumber of cores (top)\nNumber of particle pairs per core (bottom)")
plt.ylabel("Total time taken for integration (s)")


ymin = min( ppmd_gpu[1], lmps_gpu[1], min(ppmd[::,1]), min(lmps[::,1]), min(dlpoly[::, 1]) )
ymax = max( ppmd_gpu[1], lmps_gpu[1], max(ppmd[::,1]), max(lmps[::,1]), max(dlpoly[::, 1]) )



plt.ylim([ymin,ymax])

labels = [int(l) for l in ppmd[::,0]]

npp = (2./3.) * math.pi * ((float(_sim_rec[3]) * 1.1) ** 3.0) * float(_sim_rec[2])

for lx, l in enumerate(labels):
    nn = (float(_sim_rec[0]) / float(l)) * npp
    labels[lx] = str(l) + "\n" + " %1.e " % nn



ax.yaxis.grid(True, which='major')


plt.xticks(ppmd[::,0], labels)

plt.show()








