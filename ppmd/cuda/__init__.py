
__all__ = [
    'cuda_config',
    'cuda_build',
    'cuda_runtime',
    'cuda_base',
    'cuda_data',
    'cuda_cell',
    'cuda_halo',
    'cuda_loop',
    'cuda_pairloop',
    'cuda_state',
    'cuda_mpi',
    'cuda_domain'
]

CUDA_IMPORT = False
CUDA_IMPORT_ERROR = None

import cuda_config

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
    import cuda_domain
    CUDA_IMPORT = True
except Exception as e:
    CUDA_IMPORT_ERROR = e
    #print e



