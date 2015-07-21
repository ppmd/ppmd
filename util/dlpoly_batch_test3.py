#!/usr/bin/python


import os


size = range(7,19)

for s in size:
    cmd = "./dlpoly_test_run_restart.py -K %(K)s " % {'K':s}
    os.system(cmd)




































