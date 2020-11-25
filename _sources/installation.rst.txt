.. contents::



Installation
============

PPMD is developed to run on Linux 64bit machines. Other operating systems may work but are not actively developed upon at this time. 


Dependencies and Installation 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Required System tools:

* Python 3+ (``PYTHONHASHSEED`` must be set to the same non-zero value on all MPI hosts).
* mpi4py compatible MPI3.0+ (with development headers e.g. ``libopenmpi-dev openmpi-bin mpi-default-bin`` on ubuntu 17.04.).
* A C/C++ compiler, preferably intel (see ``PPMD_CC_MAIN`` environment variable).
* CUDA Toolkit if targeting CUDA.

The recommended installation method (and upgrade method) is to create a Python virtual environment then install (or upgrade) PPMD with:

``pip install --upgrade --no-cache-dir git+https://github.com/ppmd/ppmd@master``

For CUDA support please install PyCUDA:

``pip install --no-cache-dir pycuda``

The installation can then by customised with the environment variables defined below.


Environment Variables
---------------------
The code generation system relies on consistency of the following environment variable across all MPI processes, this value must be set for parallel MPI execution.

* ``PYTHONHASHSEED``: e.g ``export PYTHONHASHSEED=123``.

Set the following environment variables to alter compilers used and default temporary directories. If these variables are not set the default behaviour is to use GCC and build temporary files to ``/tmp/build``.

* ``PPMD_BUILD_DIR``: The directory used as a storage location for generated libraries. For example: ``export PPMD_BUILD_DIR=/tmp/build``
* ``PPMD_CC_MAIN``: Name of the compiler to use from compilers defined in ``ppmd/config/compilers`` or in a directory given by ``PPMD_EXTRA_COMPILERS``. For example: ``export PPMD_CC_MAIN=ICC``
* ``PPMD_CC_OMP``: Name of the OpenMP compiler to use, as above. For example: ``export PPMD_CC_OMP=$PPMD_CC_MAIN``
* ``PPMD_EXTRA_COMPILERS``: Directory that should be parsed for user defined compilers.
* ``PPMD_DISABLE_CUDA``: Disable CUDA initialisation.


.. _cuda:

The following should be set if CUDA support is desired.

* ``CUDA_SDK``: location of CUDA_SDK (or path containing the helper header files from the sdk).
* ``MPI_HOME``: Used by the default CUDA compiler configuration to locate the desired MPI implementation.


















