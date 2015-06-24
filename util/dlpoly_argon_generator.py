#!/usr/bin/python
import math
import numpy as np
import random

N       = 1000
rho     = 3.
sigma   = 3.405
eps     = 0.9661
cutoff  = 7.5
mass    = 39.948
temp    = 85.0
pressure= 0.0
steps   = 15000
steps_eq= 1000
extent  = [30.000000000, 30.000000000, 30.000000000]

#####################################################################################
#Create FIELD FILE

FIELD_STR = '''Argon
UNITS kJ
MOLECULES 1
Argon Atoms
NUMMOLS %(NUMMOLS)s
ATOMS   1
Ar         %(MASS)s    0.000000
FINISH
VDW     1
Ar      Ar      lj   %(EPS)s      %(SIGMA)s
CLOSE
''' % {'NUMMOLS':N, 'MASS':mass, 'SIGMA':sigma, 'EPS':eps}

fh = open('FIELD','w')
fh.write(FIELD_STR)
fh.close()


#####################################################################################
#Create CONTROL file

CTRL_STR = '''Argon System
temperature        %(TEMP)s
pressure	       %(PRESSURE)s
steps		       %(STEPS)s
equilibration	   %(STEPS_EQ)s
multiple           1
scale		       5
print		       100
stack		       100
stats		       10
trajectory	       1000         5         0
rdf		           10
timestep	       1.0000E-03
cutoff		       %(CUTOFF)s
delr               1.0000E+00
rvdw               %(CUTOFF)s
no electrostatics
shake tolerance	       1.0000E-04
quaternion tolerance   1.0000E-04
print rdf
job time	       3.6000E+03
close time	       1.0000E+02
finish
''' % {'TEMP':temp, 'PRESSURE':pressure, 'STEPS':steps, 'STEPS_EQ':steps_eq, 'CUTOFF':cutoff}

fh = open('CONTROL','w')
fh.write(CTRL_STR)
fh.close()

#####################################################################################
#Create CONFIG file

CFG_STR = '''Argon
        2       1       %(N)s 
        %(E0)s        0.0000000000        0.0000000000
        0.0000000000        %(E1)s        0.0000000000
        0.0000000000        0.0000000000        %(E2)s
''' % {'N':N, 'E0':extent[0],'E1':extent[1], 'E2':extent[2]}

fh = open('CONFIG','w')
fh.write(CFG_STR)

np1_3 = N**(1./3.)
np2_3 = np1_3**2.
mLx_2 = (-0.5 * extent[0]) + (0.5*extent[0])/math.floor(np1_3)

dev = 10.

for ix in range(N):
    z=math.floor(ix/np2_3)
    _tx = mLx_2+(math.fmod((ix - z*np2_3),np1_3)/np1_3)*extent[0]
    _ty = mLx_2+(math.floor((ix - z*np2_3)/np1_3)/np1_3)*extent[0]
    _tz = mLx_2+(z/np1_3)*extent[0]
    
    fh.write('Ar \t' + str(ix+1)+'\n')
    fh.write('\t' + str(_tx) + '\t' + str(_ty) + '\t' + str(_tz) + '\n')
    fh.write('\t' + str(random.uniform(-1.*dev, dev)) + '\t' + str(random.uniform(-1.*dev, dev)) + '\t' + str(random.uniform(-1.*dev, dev)) + '\n')
    fh.write('\t' + str(0.) + '\t' + str(0.) + '\t' + str(0.) + '\n')


fh.close()



















