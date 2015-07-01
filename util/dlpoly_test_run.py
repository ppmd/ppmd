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

print "--------------------------------------------------------------"



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

print "N =", N

#create DLPOLY config scripts
os.system("./dlpoly_argon_generator.py -N %(N)s" % {'N':N})

#run DL_POLY
os.system("DLPOLY.Z")



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
print "DL_POLY time taken ",int_time, "s"





WD=os.getcwd()

os.chdir("../src")

out=subprocess.check_output("./argon_example.py -N %(N)s | grep integrate" % {'N':N}, shell=True)


os.chdir(WD)

print "PPMD", out

int_time0 = re.findall('taken:(.*?)s', out)
int_time0 = float(int_time0[0])











































