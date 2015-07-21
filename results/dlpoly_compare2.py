#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt



#number of particles
N = np.array([
        4600,
        4700,
        4800,
        4900,
        4950,
        5000,
        5200,
        5400
        ])

#dl_poly results
dlpoly = np.array([
            5.63406705856,
            5.8195950985,
            5.96973896027,
            6.16937208176,
            6.42620396614,
            6.41447877884,
            6.36420989037,
            6.82895588875
          ])

#python results
ppmd = np.array([
        5.12434101105,
        5.32838106155,
        5.45761013031,
        5.71333909035,
        5.22045397758,
        5.29048991203,
        5.62258410454,
        5.99724078178
        ])



plt.plot(N, dlpoly/N, 'r', N, ppmd/N, 'b')

plt.title('DL_POLY, red, portable MD, blue')
plt.xlabel('Number of particles')
plt.ylabel('Time taken per particle (s)')

plt.show()











