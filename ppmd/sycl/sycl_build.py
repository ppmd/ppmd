from ppmd.lib import compiler, build
import os
from ppmd.sycl import sycl_config
from ppmd import config

CC = config.COMPILERS[os.environ.get("PPMD_CC_SYCL", "DPCPP")]

SYCL_HEADER = """
#include <CL/sycl.hpp>
"""

def sycl_simple_lib_creator(header_code, src_code, name=""):
    header_code = SYCL_HEADER + "\n" + header_code
    return build.simple_lib_creator(header_code, src_code, CC=CC, name=name, prefix="SYCL")

def build_static_libs(lib):
    with open(os.path.join(sycl_config.LIB_DIR, lib + ".hpp")) as fh:
        hsrc = fh.read()
    with open(os.path.join(sycl_config.LIB_DIR, lib + ".cpp")) as fh:
        src = fh.read()
    return sycl_simple_lib_creator(hsrc, src, lib)
