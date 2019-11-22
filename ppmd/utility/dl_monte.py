from __future__ import print_function, division
import numpy as np
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"


from ppmd.utility.dl_poly import get_field_value, read_field, read_domain_extent


def read_control(filename=None):
    """
    Read a DL POLY CONTROL file and return a list of the values found.
    :param filename:
    :return:
    """

    r = []

    with open(filename) as fh:
        for line_num, line in enumerate(fh):
            line = line.strip()

            if line.startswith('#') or len(line) == 0:
                continue

            else:
                r.append(
                    line.split()
                )

    return r


def get_control_value(src=None, key=None):
    if type(src) is str:
        src = read_control(src)
    key = key.lower().split()
    for kx in range(len(key)):
        src = [v[1::] for v in src if v[0].lower() == key[kx]]

    return src


def read_positions(filename):
    """
    Read positions and types from a dl_monte config containing only positions.
    """

    shift = 6

    data = []
    types = []
    with open(filename) as fh:
        for i, line in enumerate(fh):
            if i > shift:
                ls = line.strip().split()
                if len(ls) == 2:
                    types.append(' '.join(ls))
                elif len(ls) == 3:
                    data.append([float(lx) for lx in ls])
                else:
                    raise RuntimeError('Unknown line in CONFIG')
    
    return np.array(data), np.array(types)





