import os

from ppmd import config, runtime
from ppmd.cuda import cuda_config
from ppmd.lib import build

NVCC = config.COMPILERS[cuda_config.CUDA_CFG['cc-main'][1]]


def simple_lib_creator(header_code, src_code, name):
    # use the main build toolchain but with the nvidia compiler
    return build.simple_lib_creator(
        header_code, src_code, name, extensions=('.h', '.cu'), CC=NVCC,
        prefix='CUDA', inc_dirs=(runtime.LIB_DIR, cuda_config.LIB_DIR)
    )

def build_static_libs(lib):
    with open(os.path.join(cuda_config.LIB_DIR, lib+ '.h')) as fh:
        hsrc = fh.read()
    with open(os.path.join(cuda_config.LIB_DIR, lib+ '.cu')) as fh:
        src = fh.read()
    return simple_lib_creator(hsrc, src, lib)
