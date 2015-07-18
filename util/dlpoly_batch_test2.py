#!/usr/bin/python


import os


size = range(15,19)

for s in size:
    cmd = "./dlpoly_test_run_2_4.py -K %(K)s " % {'K':s}
    os.system(cmd)




































