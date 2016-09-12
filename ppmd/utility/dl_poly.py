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

def read_forces(filename=None):
    """
    Read forces from DL_POLY config.
    :arg str filename: File name of CONFIG file.
    :returns: numpy array of forces read from config.
    """

    fh = open(filename)
    shift = 9
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
            elif line == 'finish':
                break

            else:
                r.append(
                    line.split()
                )

    return r


def get_control_value(src=None, key=None):
    if type(src) is str:
        src = read_control(src)
    key = key.lower().split()
    for kx in xrange(len(key)):
        src = [v[1::] for v in src if v[0].lower() == key[kx]]

    return src



def read_field(filename=None):
    """
    Read a DL POLY FIELD file and return a dict of the values found.
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

def get_field_value(src=None, key=None):
    if type(src) is str:
        src = read_control(src)
    key = key.lower().split()
    for kx in xrange(len(key)):
        src = [v[1::] for v in src if v[0].lower() == key[kx]]

    return src




