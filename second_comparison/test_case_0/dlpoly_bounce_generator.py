#!/usr/bin/python
import math
import numpy as np
import random
import getopt
import sys
import os


N       = 2
N_e     = 1000
I       = 100000
E       = 0
R       = False
K       = 6
sep     = 0.4

rho     = 0.05
sigma   = 1.0 #3.405
eps     = 1.0 #0.9661
cutoff  = 7.5
mass    = 39.948
temp    = 85.0
pressure= 0.0
steps   = I
steps_eq= E
steps_scale = steps+1
print_steps = steps
#extent  = [30.000000000, 30.000000000, 30.000000000]
extent = [(float(N_e) / float(rho))**(1./3.),(float(N_e) / float(rho))**(1./3.),(float(N_e) / float(rho))**(1./3.)]
#extent = [float(K*cutoff),float(K*cutoff),float(K*cutoff)]


print extent
print math.ceil(extent[0]/cutoff)

#####################################################################################
#Create FIELD FILE
if (R==False):
    FIELD_STR = '''Argon
    UNITS internal
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
multiple           %(STEPS)s
scale		       %(STEPS_SCALE)s
print		       %(PRINT)s
stack		       0
stats		       %(STEPS)s
trajectory	       %(STEPS)s         5         0
rdf		           %(STEPS)s
timestep	       1.0000E-04
cutoff		       %(CUTOFF)s
delr               1.0000E+00
rvdw               %(CUTOFF)s
mxquat 0
mxshak 0
vdw shift
no electrostatics
no vafaveraging
restart noscale
mxquat 0
mxshak 0
shake tolerance	       1.0000E-04
quaternion tolerance   1.0000E-04
job time	       3.6000E+03
close time	       1.0000E+02
finish
''' % {'TEMP':temp, 'PRESSURE':pressure, 'STEPS':steps, 'STEPS_EQ':steps_eq, 'CUTOFF':cutoff, 'STEPS_SCALE':str(steps_scale), 'PRINT':print_steps}

fh = open('CONTROL','w')
fh.write(CTRL_STR)
fh.close()

#####################################################################################
#Create CONFIG file

if (R==False):

    CFG_STR = '''Argon
            2       1       %(N)s 
            %(E0)s        0.0000000000        0.0000000000
            0.0000000000        %(E1)s        0.0000000000
            0.0000000000        0.0000000000        %(E2)s
    ''' % {'N':N, 'E0':extent[0],'E1':extent[1], 'E2':extent[2]}

    fh = open('CONFIG','w')
    fh.write(CFG_STR)

    fh.write('Ar        ' + str(1)+'\n')
    fh.write('\t' + '%1.11f' % (-0.5*sep) + '\t' + '%1.11f' % (0.0) + '\t' + '%1.11f' % (0.0) + '\n')
    fh.write('\t' + '%1.11f' % (0.0) + '\t' + '%1.11f' % (0.0) + '\t' + '%1.11f' % (0.0) + '\n')
    fh.write('\t' + '%1.11f' % (0.0) + '\t' + '%1.11f' % (0.0) + '\t' + '%1.11f' % (0.0) + '\n')


    fh.write('Ar        ' + str(2)+'\n')
    fh.write('\t' + '%1.11f' % (0.5*sep) + '\t' + '%1.11f' % (0.0) + '\t' + '%1.11f' % (0.0) + '\n')
    fh.write('\t' + '%1.11f' % (0.0) + '\t' + '%1.11f' % (0.0) + '\t' + '%1.11f' % (0.0) + '\n')
    fh.write('\t' + '%1.11f' % (0.0) + '\t' + '%1.11f' % (0.0) + '\t' + '%1.11f' % (0.0) + '\n')

    fh.close()



















