from __future__ import print_function, division, absolute_import

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import numpy as np

def wrap_positions(extent, positions):
    """
    Ensure that the passed positions are contained within a domain of size
    extent with a centred origin. Returns a new array with wrapped and 
    centred positions.

    :arg extent: 3 entry subscriptable object of domain extents.
    :arg positions: Nx3 numpy array of positions (not modified).
    """

    if len(positions.shape) != 2:
        raise RuntimeError("Expected Nx3 shaped array")
    if positions.shape[1] != 3:
        raise RuntimeError("Expected Nx3 shaped array")

    out = np.zeros_like(positions)
    
    offset = np.min(positions, axis=0)

    for cx in range(3):
        # map onto [0, E_cx]
        out[:, cx] = np.fmod(positions[:, cx] - offset[cx], extent[cx])
        #min value is 0 from the offset used
        max_cx = np.max(out[:, cx])
        pad_cx = extent[cx] - max_cx
        assert pad_cx >= 0.0
        out[:, cx] -= 0.5*(extent[cx] - pad_cx)

    for cx in range(3):
        if np.min(out[:, cx]) < -0.5*extent[cx]:
            raise RuntimeError('Wrapping positions failed.')
        if np.max(out[:, cx]) > 0.5*extent[cx]:
            raise RuntimeError('Wrapping positions failed.')

    return out


