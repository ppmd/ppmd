####################################################################################################
# AccessType class
####################################################################################################


class AccessType(object):
    """
    Class to hold an access descriptor for data. In a pyop2 style manner. (WIP)

    """
    _modes = ["R", "W", "RW", "INC"]

    def __init__(self, mode):
        self._mode = mode

    def __str__(self):
        return self._mode

    @property
    def mode(self):
        return self._mode


R = AccessType("R")
W = AccessType("W")
RW = AccessType("RW")
INC = AccessType("INC")
