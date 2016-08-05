import numpy as np

def read_domain_extent(filename=None):
    """
    Read domain extent from DL_POLY CONFIG file.
    :arg str filename: File name of CONFIG file.
    :returns: numpy array of domain extent.
    """
    fh = open(filename)

    extent = np.array([0., 0., 0.])

    for i, line in enumerate(fh):
        if i == 2:
            extent[0] = line.strip().split()[0]
        if i == 3:
            extent[1] = line.strip().split()[1]
        if i == 4:
            extent[2] = line.strip().split()[2]
        else:
            pass

    fh.close()

    return extent


def read_positions(filename=None):
    """
    Read positions from DL_POLY config.
    :arg str filename: File name of CONFIG file.
    :returns: numpy array of positions read from config.
    """

    fh = open(filename)
    shift = 7
    offset = 4
    _n = 0

    data = []

    for i, line in enumerate(fh):
        if (i > (shift - 2)) and ((i - shift + 1) % offset == 0):
            _t = (float(line.strip().split()[0]),
                  float(line.strip().split()[1]),
                  float(line.strip().split()[2]))
            data.append(_t)

    fh.close()
    return np.array(data)



def read_velocities(filename=None):
    """
    Read velocities from DL_POLY config.
    :arg str filename: File name of CONFIG file.
    :returns: numpy array of velocities read from config.
    """

    fh = open(filename)
    shift = 8
    offset = 4
    _n = 0

    data = []

    for i, line in enumerate(fh):
        if (i > (shift - 2)) and ((i - shift + 1) % offset == 0):
            _t = (float(line.strip().split()[0]),
                  float(line.strip().split()[1]),
                  float(line.strip().split()[2]))
            data.append(_t)

    fh.close()
    return np.array(data)










