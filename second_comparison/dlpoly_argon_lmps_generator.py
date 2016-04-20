#!/usr/bin/python
import math
import numpy as np
import random
import getopt
import sys
import os
import hashlib


N       = 10**3 # Number of particles
I       = 100   # Number of interations
R       = 0.05  # density
C       = 2.5   # cutoff

# random seed for reproducibility
S_len = 8
S       = hashlib.sha256(str(random.random())).hexdigest()[0:S_len]

F = False       # reload config from file
F_file = None


H = False       # print help and quit





try:
    opts, args = getopt.getopt(sys.argv[1:], "hN:I:R:S:C:F:")
except getopt.GetoptError as err:
    # print help information and exit:
    print str(err) # will print something like "option -a not recognized"
    sys.exit(2)
for o, a in opts:
    if o == "-N":
        N=int(a)
    elif o == "-I":
        I=int(a)
    elif o == "-R":
        R=float(a)
    elif o == "-C":
        C=float(a)
    elif o == "-S":
        S=str(a)
    elif o == "-h":
        H = True
    elif o == "-F":
        F = True
        F_file=str(a)
        print "Loading values from file:", F_file
    else:
        assert False, "unhandled option"

if H:
    _h_str = '''
    Configuration file generation for DL_POLY, LAMMPS and PPMD.
    
    -N      Number of particles
    -I      Number of iterations
    -R      Density
    -C      LJ interaction cutoff
    -S      Random seed for velocities

    -F      File to load config values from seperated by
            whitespace in the order of the above options.

    -h      Show this help
    '''
    print _h_str
    quit()


if F:
    fh = open(F_file, 'r')
    F_str = fh.read()
    fh.close()
    
    F_str = F_str.splitlines()

    N = int(F_str[0])

    print F_str[0]

    I = int(F_str[1])
    R = float(F_str[2])
    C = float(F_str[3])
    S = str(F_str[4])




S = S.upper()
assert len(S) == S_len, "Seed has incorrect length"

# write config values to file for record

print N, I, R, C, S


fh = open('SIMULATION_RECORD', 'w')
fh.write(str(N)+'\n')
fh.write(str(I)+'\n')
fh.write(str(R)+'\n')
fh.write(str(C)+'\n')
fh.write(str(S)+'\n')
fh.close()





# set random seed
random.seed(S)




rho     = R
sigma   = 1.0 #3.405
eps     = 1.0 #0.9661
cutoff  = C
mass    = 39.948
temp    = 85.0
pressure= 0.0
steps   = I
steps_eq= 0
steps_scale = steps+1
print_steps = steps
extent = [(float(N) / float(rho))**(1./3.),(float(N) / float(rho))**(1./3.),(float(N) / float(rho))**(1./3.)]


print extent
print math.ceil(extent[0]/cutoff)

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
units lj

variable Ni equal   %(STEPS)s

pair_style lj/cut %(CUTOFF)s

processors * * * grid numa
read_data        lammps_data.lmps

pair_coeff 1 1 %(EPS)s %(SIGMA)s
pair_modify shift yes


mass 1 %(MASS)s

neighbor 0.1 bin
neigh_modify delay 0 every 10 check no




timestep 0.001



fix     1 all nve

run %(STEPS)s
write_dump all xyz lammps_out.xyz

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

















