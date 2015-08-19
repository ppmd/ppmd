####################################################################################################
# AccessType class
####################################################################################################


class AccessType(object):
    '''
    Class to hold an access descriptor for data. In a pyop2 style manner. (WIP)

    '''
    _modes = ["R", "W", "RW"]

    def __init__(self, mode):
        self._mode = mode

R = AccessType("R")
W = AccessType("W")
RW = AccessType("RW")