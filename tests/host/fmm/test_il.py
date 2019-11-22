from itertools import product

from ppmd.coulomb import fmm_interaction_lists






def test_cube():
    # should reproduce the cubic fmm interaction lists

    extent = (8,8,8)

    l, _ = fmm_interaction_lists.compute_interaction_lists(extent)
    
    
    children = (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    )


    for cl in range(8):
        ll = l[cl]
        lls = set(ll)

        assert len(lls) == len(ll)
        assert len(lls) == 189

        c = children[cl]
        

        tcount = 0
        for px in product(
            range(-2 - c[0], 4 - c[0]),
            range(-2 - c[1], 4 - c[1]),
            range(-2 - c[2], 4 - c[2]),
        ):
            apx = [abs(pxx) for pxx in px]
            if max(apx) < 2:
                continue

            px = tuple(px)

            lls.remove(px)

            tcount += 1
        
        assert tcount == 189
        assert len(lls) == 0


