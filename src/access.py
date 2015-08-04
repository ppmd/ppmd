class AccessType(object):
    _modes = ["R", "W", "RW"]

    def __init__(self, mode):
        self._mode = mode

R = AccessType("R")
W = AccessType("W")
RW = AccessType("RW")

