
from itertools import product
from math import ceil

def compute_interaction_lists(extent, subdivision=(2,2,2), c=(4/(3**0.5))-1.0000001):
    """
    Returns the interaction lists for child cells, numbered lexicographically, for a simulation domain
    extent and a subdivision rule.

    :arg extent: Domain extent in x, y, z.
    :arg subdivision: Number of subdivides that a parent is divided into in x,y,z.
    :arg c: Value of c to respect for interaction lists for rho > (c+1) a in 3D FMM MTL error bounds (default=(4/sqrt(3)) - 1)).
    """

    e = extent
    s = subdivision
    a = ((e[0]**2 + e[1]**2 + e[2]**2)**0.5) * 0.5

    rho = (c + 1) * a

    excl = (
        int(ceil(rho / e[0])) - 1,
        int(ceil(rho / e[1])) - 1,
        int(ceil(rho / e[2])) - 1,
    )

    assert excl[0] > 0
    assert excl[1] > 0
    assert excl[2] > 0
    
    # convert offsets to list of tuples that are too close
    excl_tuples = [
        tuple(px) for px in product(
            range(-excl[0], excl[0]+1),
            range(-excl[1], excl[1]+1),
            range(-excl[2], excl[2]+1),
        )
    ]
    
    # compute interaction lists for each child cell of a parent (as offsets)
    il = []
    for c2 in range(s[2]):
        for c1 in range(s[1]):
            for c0 in range(s[0]):
                
                il.append(
                    tuple([
                        tuple(px) for px in product(
                            range(-(excl[0] * s[0] + c0), excl[0] * s[0] + (s[0] - c0)),
                            range(-(excl[1] * s[1] + c1), excl[1] * s[1] + (s[1] - c1)),
                            range(-(excl[2] * s[2] + c2), excl[2] * s[2] + (s[2] - c2)),
                        ) if tuple(px) not in excl_tuples
                    ])
                )
    
    return tuple(il), excl_tuples







