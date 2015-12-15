#!/usr/bin/python


import numpy as np

from ppmd.cuda import *
 

 cuda_runtime.cuda_mem_get_info()

 d_a1 = cuda_base.Array(np.array(range(10)))
 d_a2 = cuda_base.Array(ncomp=1000)


 d_b = cuda_base.Matrix(initial_value=np.array([[1, 2], [3, 4]]))
 d_c = cuda_base.Matrix(nrow=10000, ncol=10000)
