#!/usr/bin/python

'''
Script to automate comparisons between DL_POLY and PPMD, specify number of atoms
using "   -N <number of atoms>    " .
'''

import math
import numpy as np
import random
import getopt
import sys
import re
import os
import subprocess
import timeit
import time
import datetime

print "--------------------------------------------------------------"



N       = 10**3
I       = 5000
NP      = 2
E       = 100
K       = 6

try:
    opts, args = getopt.getopt(sys.argv[1:], "N:I:P:K:")
except getopt.GetoptError as err:
    # print help information and exit:
    print str(err) # will print something like "option -a not recognized"
    sys.exit(2)
for o, a in opts:
    if o == "-N":
        N=int(a)
    elif o == "-I":
        I=int(a)
    elif o == "-P":
        NP=int(a)
    elif o== "-K":
        K=int(a)
        N=(K**3)*23
    else:
        assert False, "unhandled option"

print "N =", N
print "I =", I

#create DLPOLY EQULIBRIUM config scripts
#os.system("./dlpoly_argon_generator.py -N %(N)s -E %(E)s" % {'N':N, 'E':E})
os.system("./dlpoly_argon_generator.py -K %(K)s -E %(E)s" % {'K':K, 'E':E})
#run DL_POLY
cmd = "mpirun -n %(NP)s DLPOLY.Z" % {'NP': 4}
os.system(cmd)



#create DLPOLY config scripts for main run
#os.system("./dlpoly_argon_generator.py -N %(N)s -I %(I)s -R" % {'N':N, 'I':I})
os.system("./dlpoly_argon_generator.py -K %(K)s -I %(I)s -R" % {'K':K, 'I':I})

#run DL_POLY
cmd = "mpirun -n %(NP)s DLPOLY.Z" % {'NP': NP}
t1 = timeit.timeit(stmt='os.system(cmd)', setup='from __main__ import os, cmd', number=1)

print "DL_POLY total time", t1, "s"

#Get DL_POLY time taken
fh=open('OUTPUT')
line_count = -1
int_time = -1
for i, line in enumerate(fh):
    items=re.findall("cpu",line,re.MULTILINE)
    if (len(items)>0):
        line_count = i+5
    elif (i==line_count):
        if(len(line.strip().split())>0):
            int_time = line.strip().split()[0]
fh.close()
print "DL_POLY integrate time taken:",int_time, "s"



WD=os.getcwd()

os.chdir("../src")

'''For testing'''
#cmd0 = "mpirun -n %(NP)s ./argon_example.py -N %(N)s -I %(I)s" % {'NP': NP, 'N':N, 'I':I}
#os.system(cmd0)



cmd0 = "mpirun -n %(NP)s ./argon_example.py -N %(N)s -I %(I)s | grep integrate" % {'NP': NP, 'N':N, 'I':I}



t0_0 = time.time()
out=subprocess.check_output(cmd0, shell=True)
t0_1 = time.time()
t0 = t0_1 - t0_0

#t0 = timeit.timeit(stmt='out=subprocess.check_output(cmd0, shell=True)', setup='from __main__ import subprocess, cmd0', number=1)



os.chdir(WD)
print "-"
print "PPMD total time", t0, "s" 
print "PPMD", out

int_time0 = re.findall('taken:(.*?)s', out)
int_time0 = float(int_time0[0])



fh = open('compare.txt', 'a+')

#_date=datetime.datetime.now().date()

'''NP \t N \t I \t DLPOLY_TOTAL \t PPMD_TOTAL '''

fh.write("%(NP)s\t%(N)s\t%(I)s\t%(DLPOLY_TOTAL)s\t%(PPMD_TOTAL)s\t%(DLPOLY)s\t%(PPMD)s\n" % {'NP': NP, 'N':N, 'I':I, 'DLPOLY_TOTAL':t1, 'PPMD_TOTAL':t0, 'DLPOLY':int_time, 'PPMD':int_time0, })
fh.close()





































