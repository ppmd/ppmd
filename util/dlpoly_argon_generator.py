#!/usr/bin/python
import math
import numpy as np
import random
import getopt
import sys



N       = 10**3

try:
    opts, args = getopt.getopt(sys.argv[1:], "N:")
except getopt.GetoptError as err:
    # print help information and exit:
    print str(err) # will print something like "option -a not recognized"
    sys.exit(2)
for o, a in opts:
    if o == "-N":
        N=int(a)          
    else:
        assert False, "unhandled option"



rho     = 0.05
sigma   = 3.405
eps     = 0.9661
cutoff  = 7.5
mass    = 39.948
temp    = 85.0
pressure= 0.0
steps   = 5000
steps_eq= 0
steps_scale = steps+1
print_steps = steps
#extent  = [30.000000000, 30.000000000, 30.000000000]
extent = [(float(N) / float(rho))**(1./3.),(float(N) / float(rho))**(1./3.),(float(N) / float(rho))**(1./3.)]

#print extent

#####################################################################################
#Create FIELD FILE

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
timestep	       1.0000E-03
cutoff		       %(CUTOFF)s
delr               1.0000E+00
rvdw               %(CUTOFF)s
no electrostatics
no vafaveraging
restart noscale
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

dev = .2

for ix in range(N):
    z=math.floor(ix/np2_3)
    _tx = mLx_2+(math.fmod((ix - z*np2_3),np1_3)/np1_3)*extent[0]
    _ty = mLx_2+(math.floor((ix - z*np2_3)/np1_3)/np1_3)*extent[0]
    _tz = mLx_2+(z/np1_3)*extent[0]
    
    fh.write('Ar        ' + str(ix+1)+'\n')
    fh.write('\t' + str(_tx) + '\t' + str(_ty) + '\t' + str(_tz) + '\n')
    fh.write('\t' + '%1.11f' % (random.uniform(-1.*dev, dev)) + '\t' + '%1.11f' % (random.uniform(-1.*dev, dev)) + '\t' + '%1.11f' % (random.uniform(-1.*dev, dev)) + '\n')
    fh.write('\t' + '0.00000000000' + '\t' + '0.0000000000' + '\t' + '0.00000000000' + '\n')


fh.close()



















