#!/usr/bin/python

import numpy as np

fh=open('compare.txt','r')

data=[]

for i, line in enumerate(fh):
    d=[]
    for j in line.strip().split():
        d.append(float(j))
    data.append(d)
    
data = np.array(data)







fh.close()
