#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt



#number of particles
N = np.array([10**3,
        11**3,
        12**3,
        13**3,
        14**3,        
        15**3,
        16**3, 
        17**3,
        18**3,
        19**3,
        20**3,
        21**3,
        22**3,
        23**3
        ])

#dl_poly results
dlpoly = np.array([12.523,
          16.121,
          20.367,
          25.284,
          31.830,
          37.937,
          46.316,
          55.489,
          64.869,
          76.026,
          88.226,
          101.402,
          117.960,
          134.457
          ])

#python results
ppmd = np.array([19.5531289577,
        29.6130080223,
        24.6790940762,
        35.0785570145,
        32.166916132,
        43.1939308643,
        57.7676410675,
        54.2102949619,
        70.3537499905,
        90.0300419331,
        84.406393528,
        105.961280823,
        134.046503067,
        122.972121954
        ])

print N

mask = [0,2,4,7,10,13]
print ppmd[mask]
print N[mask]
print ppmd[mask]/N[mask]

plt.plot(N, dlpoly/N, 'r', N[mask], ppmd[mask]/N[mask], 'b')

plt.title('DL_POLY, red, portable MD, blue')
plt.xlabel('Number of particles')
plt.ylabel('Time taken per particle (s)')

plt.show()











