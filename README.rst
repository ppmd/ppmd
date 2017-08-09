


Introduction
------------

<Project Name> is a portable high level framework to create high performance Molecular Dynamics codes. The principle idea is that a simulation consists of sets of particles and most operations on these particles can be described using either a loop over all particles or a loop over particle pairs and applying some operation.


Installation
------------

The contents of this git repository should be placed somewhere found by the ``PYTHONPATH`` environment variable.


Dependencies
~~~~~~~~~~~~

Required System tools:

* Python 2.7.8 (Tests pass on Python 3.4)
* mpi4py compatible MPI
* A C compiler, preferably intel
* CUDA Toolkit if targeting CUDA.

To pip install python packages:
``pip install --no-cache-dir -r requirements.txt"

For CUDA support:
``pip install --no-cache-dir -r requirements_cuda.txt"


Environment Variables
---------------------
Set the following environment variables.

* ``MPI_HOME``: Used by the default CUDA and MPI compiler configurations to locate the desired MPI implementation.
* ``BUILD_DIR``: The directory used as a storage location for generated libraries. For example: ``mkdir /tmp/build; export BUILD_DIR=/tmp/build``
* ``CUDA_SDK``: location of CUDA_SDK (or path containing the helper header files form the sdk).

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






