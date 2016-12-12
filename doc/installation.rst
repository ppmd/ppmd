.. contents::



Installation
============

PPMD is developed to run on Linux 64bit machines. Other operating systems may work but are not actively developed upon at this time. 


Git Repository
~~~~~~~~~~~~~~

https://bitbucket.org/wrs20/ppmd/src


Dependencies
~~~~~~~~~~~~

Required tools:

* Python 2.7
* mpi4py compatible MPI
* A C compiler, preferably intel
* CUDA Toolkit if targeting CUDA.


Required Python Packages

* NumPy
* MPI4Py
* ctypes
* cgen
* PyCUDA (If targeting CUDA)


Optional Python Packages

* matplotlib (if plotting is required)

To generate gpu code PPMD requires that the CUDA Toolkit is installed see further: :ref:`cuda`.



Environment Variables
---------------------

* ``MPI_HOME``: Used by the default CUDA and MPI compiler configurations to locate the desired MPI implementation.
* ``BUILD_DIR``: The directory used as a storage location for generated libraries, default ``/tmp/build``.


Compilers
~~~~~~~~~

Compliers are defined in the ``compilers`` sub-directory of the ``config_dir`` directory found in the main library directory. Future versions will support loading configurations from alternative directories. Each complier is defined in a separate file and is identified by name in the main configuration file.



.. _cuda:

CUDA
~~~~
The PPMD CUDA toolchain :mod:`~gpucuda` expects to find a CUDA toolkit installed at the environment variable ``CUDA_INSTALL_PATH``.
















