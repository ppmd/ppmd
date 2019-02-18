from __future__ import print_function, division, absolute_import

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level
import ctypes
import numpy as np

# package level
from ppmd.lib import build
from ppmd import mpi

REAL = ctypes.c_double
INT64 = ctypes.c_int64

MPI = mpi.MPI

class ParticleDatModifier:

    def __init__(self, dat, is_positiondat):
        self.dat = dat
        self.is_positiondat = is_positiondat

    def __enter__(self):
        return self.dat.view

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dat.mark_halos_old()
        if self.is_positiondat:
            self.dat.group.invalidate_lists = True
            # need to trigger MPI rank consistency/ownership here


class GlobalDataMover:

    def __init__(self, state):
        self.state = state
        self.comm = state.domain.comm
        
        self._recv_count = np.zeros(1, INT64)
        self._win = MPI.Win()
        self._win_recv_count = self._win.Create(self._recv_count, comm=self.comm)


    def __call__(self):
        state = self.state
        comm = self.comm
        rank = comm.rank
        topo = mpi.cartcomm_top_xyz(comm)
        dims = mpi.cartcomm_dims_xyz(comm)
        extent = state.domain.extent
        dist_cell_widths = [1.0 / (ex / dx) for ex, dx in zip(extent, dims)]
        dist_cell_widths = np.array(dist_cell_widths, dtype=REAL)
        npart = state.npart_local
        pos = state.get_position_dat()
        pos = pos.view

        
        lcount = 0
        lrank = np.zeros(npart, dtype=INT64)
        lpid = np.zeros(npart, dtype=INT64)
        lrind = np.zeros(npart, dtype=INT64)
        
        rk_offsets = (1, dims[0], dims[0]*dims[1])

        def to_mpi_rank(_p):
            _rk = 0
            for dx in range(3):
                assert _p[dx] <=  0.5 * extent[dx], "outside domain"
                assert _p[dx] >= -0.5 * extent[dx], "outside domain"
                tint = int((_p[dx] + 0.5 * extent[dx]) * dist_cell_widths[dx])
                tint = min(dims[dx], tint)
                _rk += tint * rk_offsets[dx]
            return _rk

        for px in range(npart):
            rk = to_mpi_rank(pos[px])
            if rk != rank:
                lrank[lcount] = rk
                lpid[lcount] = px
                lcount += 1
        
        self._recv_count[0] = 0
        self._win_recv_count.Fence(MPI.MODE_NOSTORE)
        for px in range(lcount):
            rk = lrank[px]
            self._win_recv_count.Get_accumulate(np.array(1), lrind[px:px+1], rk)
        self._win_recv_count.Fence(MPI.MODE_NOSTORE)
        
        nbytes = 0

        def byte_per_element(dat):
            return getattr(self.state, dat).view[0,0].nbytes
        def dat_dtype(dat):
            return getattr(self.state, dat).dtype
        def dat_ncomp(dat):
            return getattr(self.state, dat).ncomp


        for dat in self.state.particle_dats:
            nbytes += byte_per_element(dat) * dat_ncomp(dat)

        send = np.zeros((lcount, nbytes), np.byte)
        
        print("------->", lcount)

        view_dict = {}
        s = 0
        for dat in self.state.particle_dats:
            w = byte_per_element(dat)
            n = dat_ncomp(dat)
            w *= n
            view_dict[dat] = send[:, s:w:].view(dat_dtype(dat))
            s += w

        print(view_dict)



        # need to place the data in the remote buffers here


        #print("------->", lcount)
        #self.state.remove_by_slot(lpid[:lcount:])





        




