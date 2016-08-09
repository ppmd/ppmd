#TODO Move this to future looping package





class Distance(object):
    """
    Class to store Euclidean distances in a way that allows equality
    comparison.
    """

    def __init__(self, distance=0.0):
        self._d = distance

    @property
    def value(self):
        return self._d