import atexit


__all__ = [
    'cuda_build',
    'cuda_runtime',
    'cuda_base',
    'cuda_data'
]


import cuda_runtime
import cuda_build
import cuda_base
import cuda_data











#####################################################################################
# Module Init
#####################################################################################

cuda_runtime.cuda_set_device()



#####################################################################################
# Module cleanup
#####################################################################################

def gpucuda_cleanup():
    cuda_runtime.cuda_device_reset()


atexit.register(gpucuda_cleanup)




