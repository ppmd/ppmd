from __future__ import print_function, division
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

import pytest
import numpy as np
import ppmd as md
import os

dlpoly = md.utility.dl_poly

RES_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../res/'))
print(RES_DIR)
rank = md.mpi.MPI.COMM_WORLD.Get_rank()
nproc = md.mpi.MPI.COMM_WORLD.Get_size()

@pytest.mark.skip
def nacl_lattice(crn, e, sd=0.1, seed=87712846):

    raw_lattice = md.utility.lattice.cubic_lattice(crn, e)

    N = crn[0]*crn[1]*crn[2]

    charges = np.zeros(N)
    raw_labels = ('Na', 'Cl')
    labels = np.zeros(N, dtype='|S2')

    raw_charges = (1.0, -1.0)
    counts = [0, 0]
    lattice = np.zeros((N,3))

    nai = 0
    cli = -1

    for px in range(N):
        ix = px % crn[0]
        iy = ((px - ix) // crn[0]) % crn[1]
        iz = (px - ix - iy*crn[0])//(crn[0]*crn[1])

        t = (ix + iy + iz) % 2

        if t == 0:
            idx = nai
            nai += 1
        else:
            idx = cli
            cli -= 1

        charges[idx] = raw_charges[t]
        lattice[idx, :] = raw_lattice[px, :]
        labels[idx] = raw_labels[t]
        counts[t] += 1


    rng = np.random.RandomState(seed=seed)
    velocities = rng.normal(0.0, sd, size=(N,3))

    return lattice, velocities, charges, labels, counts

def test_dlpoly_config_read():

    crnxy = 6
    crnz = 6
    rhosl = 2.*3.**(1./3.)

    Exy = rhosl * crnxy
    Ez = rhosl * crnz
    N = crnxy*crnxy*crnz
    # recreate the data from the CONFIG
    data = nacl_lattice((crnxy, crnxy, crnz), (Exy, Exy, Ez))

    CFG = os.path.join(RES_DIR, 'dlpoly/CONFIG')

    extent = dlpoly.read_domain_extent(CFG)
    assert abs(extent[0] - Exy) < 10.**-10, "bad extent x"
    assert abs(extent[1] - Exy) < 10.**-10, "bad extent y"
    assert abs(extent[2] - Ez) < 10.**-10, "bad extent z"

    symbols = dlpoly.read_symbols(CFG)
    ids = dlpoly.read_ids(CFG)
    pos = dlpoly.read_positions(CFG)
    vel = dlpoly.read_velocities(CFG)
    frc = dlpoly.read_forces(CFG)
    for px in range(N):
        assert symbols[px] == data[3][px], "bad symbol name"
        assert ids[px] == px + 1, "bad atom id"
        assert abs(pos[px, 0] - data[0][px, 0]) < 10.**-11, "bad pos x"
        assert abs(pos[px, 1] - data[0][px, 1]) < 10.**-11, "bad pos y"
        assert abs(pos[px, 2] - data[0][px, 2]) < 10.**-11, "bad pos z"
        assert abs(vel[px, 0] - data[1][px, 0]) < 10.**-11, "bad vel x"
        assert abs(vel[px, 1] - data[1][px, 1]) < 10.**-11, "bad vel y"
        assert abs(vel[px, 2] - data[1][px, 2]) < 10.**-11, "bad vel z"
        assert abs(frc[px, 0] - float(px)) < 10.**-16, "bad force x"
        assert abs(frc[px, 1] - float(px+1)) < 10.**-16, "bad force x"
        assert abs(frc[px, 2] - float(px+2)) < 10.**-16, "bad force x"


def test_dlpoly_control_read():

    CONTROL = dlpoly.read_control(os.path.join(RES_DIR, 'dlpoly/CONTROL'))
    get = dlpoly.get_control_value

    assert get(CONTROL, 'ENSEMBLE')[0][0] == 'NVE', "bad single value"
    assert get(CONTROL, 'CLOSE')[0] == ['TIME', '1.0000E+02'], "bad multi value"
    assert get(CONTROL, 'FINISH')[0] == [], "bad no value"
    with pytest.raises(IndexError):
        err = get(CONTROL, 'NULL')[0]

def test_dlpoly_field_read():

    FIELD = dlpoly.read_field(os.path.join(RES_DIR, 'dlpoly/FIELD'))
    get = dlpoly.get_field_value
    assert get(FIELD, 'UNITS')[0][0] == 'EV', "bad single value"
    assert len(get(FIELD, 'Na')) == 4, "bad number of entries read"
    assert get(FIELD, 'Na')[0] == []
    assert get(FIELD, 'Na')[1] == ['22.9898', '1.0']
    assert get(FIELD, 'Na')[2] == ['Na', 'LJ', '0.13', '2.35']
    assert get(FIELD, 'Na')[3] == ['Cl', 'LJ', '0.11', '3.40']
    assert get(FIELD, 'CLOSE')[0] == []



















