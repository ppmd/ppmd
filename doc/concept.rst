.. contents::

Introduction
============


Performance Portable Molecular Dynamics (PPMD) is a portable high level framework to create high performance Molecular Dynamics codes. The principle idea is that a simulation consists of sets of particles and most operations on these particles can be described using either a loop over all particles or a loop over particle pairs and applying some operation.


Particle Data
~~~~~~~~~~~~~

Particle properties are stored within :class:`~data.ParticleDat` containers. These can be considered as two dimensional matrices with each row storing data for a particle. For example the positions of :math:`N` particles would be stored within a :math:`N` X :math:`3` ParticleDat. 



