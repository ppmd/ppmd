import data
import particle
import gpucuda


####################################################################################################
# Link class
####################################################################################################

class Link(object):
    """
    Class to link a dat with another. "host" dat with a "guest dat".

    :arg instance: Initialised Dat style class.
    :arg copy_to_handle: Function handle to execute to copy data to linked dat.
    :arg copy_from_handle: Function handle to execute to copy data from linked dat.
    :arg resize_handle: Function handle to execute to resize linked dat, should take new size as only argument.
    :arg cleanup_handle: Function handle to execute to cleanup linked dat if applicable.
    """
    def __init__(self,
                 instance=None,
                 copy_to_handle=None,
                 copy_from_handle=None,
                 resize_handle=None,
                 cleanup_handle=None):

        pass






####################################################################################################
# DatLinker class
####################################################################################################


class DatLinker(object):
    """
    Class to link multiple Data classes together and provide syncing between them.
    """
    def __init__(self):
        pass