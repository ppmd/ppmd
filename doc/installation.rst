.. contents::



Installation
============

PPMD is developed to run on Linux 64bit machines. Other operating systems may work but are not actively developed upon at this time. 

Dependencies
~~~~~~~~~~~~

Required tools:

* Python 2.7
* mpi4py compatible MPI
* A C compiler

Required Python Packages

* NumPy
* MPI4Py
* ctypes

Optional Python Packages

* matplotlib (if plotting is required)

To generate gpu code PPMD requires that the CUDA Toolkit is installed see further: :ref:`cuda`.



Build Directory
~~~~~~~~~~~~~~~

By default generated code is created and compiled in ``./build`` directory. This can be changed by setting the environment variable ``BUILD_DIR``.

Compilers
~~~~~~~~~

The system version of GCC is the compiler configured by default. Other compilers may be configured with the :class:`~build.Compiler` class.


.. _cuda:

CUDA
~~~~
The PPMD CUDA toolchain :mod:`~gpucuda` expects to find a CUDA toolkit installed at the environment variable ``CUDA_INSTALL_PATH``.
















