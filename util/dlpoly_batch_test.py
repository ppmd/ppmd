#!/usr/bin/python


import os


#sizes = range(7,19)
#iterations = [1000, 10000, 50000]

size = range(7,9)
iterations = [100,200,300,400,500]

'''nprocs'''
np=[1,2,4]

for s in size:
    for i in iterations:
        for n in np:
            cmd = "./dlpoly_test_run.py -K %(K)s -I %(I)s -P %(NP)s " % {'K':s, 'I':i, 'NP':n}
            os.system(cmd)




































