from setuptools import setup

long_description = """PPMD is a portable high level framework to create high performance Molecular Dynamics codes. The principle idea is that a simulation consists of sets of particles and most operations on these particles can be described using either a loop over all particles or a loop over particle pairs and applying some operation.
"""

install_requires = []
with open('requirements.txt') as fh:
    for l in fh:
        if len(l) > 0:
            install_requires.append(l)

setup(
   name='ppmd',
   version='1.0',
   description='Performance Portable Molecular Dynamics',
   license="GPL3",
   long_description=long_description,
   author='William R Saunders',
   author_email='W.R.Saunders@bath.ac.uk',
   url="https://bitbucket.org/wrs20/ppmd",
   packages=['ppmd'],
   install_requires=install_requires,
   scripts=[]
)
