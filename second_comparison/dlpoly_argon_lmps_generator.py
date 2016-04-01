#!/usr/bin/python
import math
import numpy as np
import random
import getopt
import sys
import os


N       = 10**3
I       = 100
E       = 0
R       = False
K       = 6
Q       = False

try:
    opts, args = getopt.getopt(sys.argv[1:], "RN:I:E:K:Q")
except getopt.GetoptError as err:
    # print help information and exit:
    print str(err) # will print something like "option -a not recognized"
    sys.exit(2)
for o, a in opts:
    if o == "-N":
        N=int(a)
    elif o == "-I":
        I=int(a)
    elif o == "-E":
        E=int(a)
    elif o == "-R":
        R=True
    elif o == "-Q":
        Q=True        
    elif o == "-K":
    	K=int(a)
        N=(K**3)*23
    else:
        assert False, "unhandled option"

if (E>I):
	I=E




rho     = 0.02
sigma   = 1.0 #3.405
eps     = 1.0 #0.9661
cutoff  = 2.5
mass    = 39.948
temp    = 85.0
pressure= 0.0
steps   = I
steps_eq= E
steps_scale = steps+1
print_steps = steps
#extent  = [30.000000000, 30.000000000, 30.000000000]
extent = [(float(N) / float(rho))**(1./3.),(float(N) / float(rho))**(1./3.),(float(N) / float(rho))**(1./3.)]
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
timestep	       1.0000E-03
cutoff		       %(CUTOFF)s
delr               1.0000E+00
rvdw               %(CUTOFF)s
mxquat 0
mxshak 0
vdw shift
no vom
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



# create lammps input file


LMP_INPUT = '''

atom_style full
units metal

variable Ni equal   %(STEPS)s

pair_style lj/cut %(CUTOFF)s

processors * * * grid numa
read_data        lammps_data.lmps

pair_coeff * * %(EPS)s %(SIGMA)s

mass 1 %(MASS)s

timestep 0.001



fix     1 all nve

run $Ni

''' % {'MASS': mass, 'CUTOFF': cutoff, 'EPS':eps, 'SIGMA':sigma, 'STEPS': steps}



fh = open('lammps_input.lmps','w')
fh.write(LMP_INPUT)
fh.close()




LMP_DATA = '''  lammps argon data

%(NATOMS)s atoms
1 atom types

%(xl)s %(xh)s xlo xhi
%(yl)s %(yh)s ylo yhi
%(zl)s %(zh)s zlo zhi

Masses

1 %(MASS)s

Atoms

''' % {'NATOMS': N,
        'xl': -0.5*extent[0], 'xh': 0.5*extent[0],
        'yl': -0.5*extent[1], 'yh': 0.5*extent[1],
        'zl': -0.5*extent[2], 'zh': 0.5*extent[2],
        'MASS': mass
}



fh = open('lammps_data.lmps','w')
fh.write(LMP_DATA)
fh.close()




#####################################################################################
#Create CONFIG file


CFG_STR =  '''Argon'''.ljust(72,' ') + '\n'
CFG_STR += ('''2 1 %(N)s'''% {'N':N}).ljust(72,' ') + '\n'
CFG_STR += ('''%(E0)s 0.0000000000 0.0000000000''' % {'E0':'%.10g' % extent[0]}).ljust(72,' ') + '\n'
CFG_STR += ('''0.0000000000 %(E1)s 0.0000000000''' % {'E1':'%.10g' % extent[1]}).ljust(72,' ') + '\n'
CFG_STR += ('''0.0000000000 0.0000000000 %(E2)s''' % {'E2':'%.10g' % extent[2]}).ljust(72,' ') + '\n'

fh = open('CONFIG','w')
fh.write(CFG_STR)

# general lammmps data
fh_lmps = open('lammps_data.lmps', 'a')

# temp lammps velocity
fh_lmps_v = open('lammps_data_vel.lmps', 'w')

fh_lmps_v.write('\nVelocities\n\n')


np1_3 = math.ceil(N**(1./3.))
np2_3 = np1_3**2.
mLx_2 = (-0.5 * extent[0]) + (0.5*extent[0])/np1_3

dev = 1.

for ix in range(N):
    z=math.floor(ix/np2_3)
    _tx = mLx_2+(math.fmod((ix - z*np2_3),np1_3)/np1_3)*extent[0]
    _ty = mLx_2+(math.floor((ix - z*np2_3)/np1_3)/np1_3)*extent[0]
    _tz = mLx_2+(z/np1_3)*extent[0]

    # dl_poly header
    fh.write(('Ar ' + str(ix+1)).ljust(72,' ') +'\n')
    
    #lmps header
    fh_lmps.write(str(ix+1) + ' 0 1 0 ')
    fh_lmps_v.write(str(ix+1) + ' ')

    # dl_poly head
    fh.write(('%.10g' % _tx + ' ' + '%.10g' % _ty + ' ' + '%.10g' % _tz).ljust(72,' ') + '\n')
    # lmps header
    fh_lmps.write('%.10g' % _tx + ' ' + '%.10g' % _ty + ' ' + '%.10g' % _tz + '\n')

    _vel_str = '%.10g' % (random.normalvariate(0, dev)) + ' ' + '%.10g' % (random.normalvariate(0, dev)) + ' ' + '%.10g' % (random.normalvariate(0, dev))


    #dl_poly write velocities
    fh.write(_vel_str.ljust(72,' ') + '\n')

    # lmps write velocity
    fh_lmps_v.write(_vel_str + '\n')

    # dl_poly force (could be avoided)
    fh.write(('0.00000000' + ' ' + '0.00000000' + ' ' + '0.00000000').ljust(72,' ') + '\n')

fh.close()
fh_lmps.close()
fh_lmps_v.close()

os.system('cat lammps_data_vel.lmps >> lammps_data.lmps')
os.system('rm lammps_data_vel.lmps')

















