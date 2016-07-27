import atexit


__all__ = [
    'cuda_build',
    'cuda_runtime',
    'cuda_base',
    'cuda_data',
    'cuda_cell',
    'cuda_halo',
    'cuda_loop',
    'cuda_pairloop',
    'cuda_state',
    'cuda_mpi'
]

CUDA_IMPORT = False

try:
    import cuda_runtime
    import cuda_build
    import cuda_base
    import cuda_data
    import cuda_cell
    import cuda_halo
    import cuda_loop
    import cuda_pairloop
    import cuda_state
    import cuda_mpi
    CUDA_IMPORT = True
except:
    pass





#####################################################################################
# Module Init
#####################################################################################

#if CUDA_IMPORT:
#    cuda_runtime.cuda_set_device()



#####################################################################################
# Module cleanup
#####################################################################################

'''
def gpucuda_cleanup():
    if CUDA_IMPORT:
        print "CUDA CLEANUP"
        cuda_runtime.cuda_device_reset()

if CUDA_IMPORT:
    atexit.register(gpucuda_cleanup)
'''



