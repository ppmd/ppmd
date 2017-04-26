

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



def numpy_to_xyz(arr, filename, symbol='A'):
    """
    Write a N*3 array to a file in xyz format
    :param arr: numpy array to write.
    :param filename: name of file to write to.
    """

    with open(filename, 'w') as fh:
        fh.writelines(str(arr.shape[0])+'\n')
        fh.writelines('Written by numpy_to_xyz.\n')
        for ix in xrange(arr.shape[0]):

            if type(symbol) is not str:
                sym = symbol[ix]
            else:
                sym = symbol

            fh.writelines(
                sym + '        {:.16f}        {:.16f}        {:.16f}\n'.format(arr[ix,0], arr[ix,1], arr[ix,2])
            )
































