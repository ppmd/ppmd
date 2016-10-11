

# system level imports
import numpy as np
import os


class XYZ(object):
    """
    Hold the data read from an xyz file.
    """
    def __init__(self, filename):

        self.num_atoms = 0
        """Number of atoms in given xyz file."""

        self.comment = ''
        """Comment provided in given xyz file."""

        labels = []
        positions = []

        with open(filename, 'r') as fh:
            for lx, line in enumerate(fh):
                if lx == 0:
                    # first line has number of atoms
                    self.num_atoms = int(line.split()[0])
                elif lx == 1:
                    # first line has number of atoms
                    self.comment = line[:-1]
                else:
                    l = line.split()

                    if len(l) > 0:

                        if len(l) == 4:
                            s = 1
                            labels += [l[0]]
                        else:
                            s = 0

                        positions +=[[
                            float(l[0+s]),
                            float(l[1+s]),
                            float(l[2+s])
                        ]]

        if len(labels) == 0:
            labels = None
        else:
            labels = np.array(labels)

        self.labels = labels
        """Atom labels in file if found"""

        self.positions = np.array(positions)
        """Atoms positions in file."""





