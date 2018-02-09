

Introduction
------------

<Project Name> is a portable high level framework to create high performance Molecular Dynamics codes. The principle idea is that a simulation consists of sets of particles and most operations on these particles can be described using either a loop over all particles or a loop over particle pairs and applying some operation.


Installation
------------

The contents of this git repository should be placed somewhere found by the ``PYTHONPATH`` environment variable.


Dependencies
~~~~~~~~~~~~

Required System tools:

* Python 3+ (``PYTHONHASHSEED`` must be set to the same non-zero value on all MPI hosts).
* mpi4py compatible MPI3.0+ (with development headers e.g. ``libopenmpi-dev openmpi-bin mpi-default-bin`` on ubuntu 17.04.).
* A C compiler, preferably intel (see ``PPMD_CC_MAIN`` environment variable).
* CUDA Toolkit if targeting CUDA.

To pip install python packages:

``pip install --no-cache-dir -r requirements.txt"

For CUDA support:

``pip install --no-cache-dir -r requirements_cuda.txt"


Environment Variables
---------------------
The code generation system relies on consistency of the following environment variable accross all MPI processes, this value must be set for parallel MPI execution.

* ``PYTHONHASHSEED``: e.g ``export PYTHONHASHSEED=123``.

Set the following environment variables to alter compilers used and default temporary directories. If these variables are not set the default behaviour is to use GCC and build temporary files to ``/tmp/build``.

* ``PPMD_BUILD_DIR``: The directory used as a storage location for generated libraries. For example: ``export PPMD_BUILD_DIR=/tmp/build``
* ``PPMD_CC_MAIN``: Name of the compiler to use from compilers defined in ``ppmd/config/compilers``. For example: ``export PPMD_CC_MAIN=ICC``
* ``PPMD_CC_OMP``: Name of the OpenMP compiler to use, as above. For example: ``export PPMD_CC_OMP=$PPMD_CC_MAIN``

The following should be set if CUDA support is desired.

* ``CUDA_SDK``: location of CUDA_SDK (or path containing the helper header files from the sdk).
* ``MPI_HOME``: Used by the default CUDA compiler configuration to locate the desired MPI implementation.

License
-------

Copyright (C) 2017 W.R.Saunders

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

``https://www.gnu.org/licenses/gpl.txt``






