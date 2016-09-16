


Introduction
------------

<Project Name> is a portable high level framework to create high performance Molecular Dynamics codes. The principle idea is that a simulation consists of sets of particles and most operations on these particles can be described using either a loop over all particles or a loop over particle pairs and applying some operation.


Installation
------------

The contents of this git repository should be placed somewhere found by the ``PYTHONPATH`` environment variable.


Dependencies
~~~~~~~~~~~~

Required System tools:

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

Environment Variables
---------------------

* ``MPI_HOME``: Used by the default CUDA and MPI compiler configurations to locate the desired MPI implementation.
* ``BUILD_DIR``: The directory used as a storage location for generated libraries.










