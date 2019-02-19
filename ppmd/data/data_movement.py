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
            self.dat.group.check_position_consistency()


class GlobalDataMover:

    def __init__(self, state):
        self.state = state
        self.comm = state.domain.comm
        
        self._recv_count = np.zeros(1, INT64)
        self._win = MPI.Win()
        self._win_recv_count = self._win.Create(self._recv_count, comm=self.comm)

        self._recv = None
        self._win_recv = None
    

    def _check_recv_win(self):
        
        nbytes = self._get_nbytes()

        # MPI Win create calls are collective on the comm
        if (self._recv is None) or \
                (self._recv.shape[0] < self._recv_count[0]) or \
                (self._recv.shape[1] != nbytes):
            realloc = True
        else:
            realloc = False
        realloc = np.array(realloc, np.bool)
        result = np.array(False, np.bool)
        self.comm.Allreduce(realloc, result, MPI.LOR)

        if realloc:
            self._recv = np.zeros((self._recv_count[0]+100, nbytes), dtype=np.byte)
            if self._win_recv is not None:
                self._win_recv.Free()
            self._win_recv = self._win.Create(self._recv, disp_unit=nbytes, comm=self.comm)


    def _byte_per_element(self, dat):
        return getattr(self.state, dat).view.itemsize
    def _dat_dtype(self, dat):
        return getattr(self.state, dat).dtype
    def _dat_ncomp(self, dat):
        return getattr(self.state, dat).ncomp
    def _dat_obj(self, dat):
        return getattr(self.state, dat)

    def _get_nbytes(self):
        nbytes = 0
        for dat in self.state.particle_dats:
            nbytes += self._byte_per_element(dat) * self._dat_ncomp(dat)
        return nbytes


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
                tint = min(dims[dx]-1, tint)
                _rk += tint * rk_offsets[dx]
            return _rk

        for px in range(npart):
            rk = to_mpi_rank(pos[px])
            if rk != rank:
                lrank[lcount] = rk
                lpid[lcount] = px
                lcount += 1
        
        self._recv_count[0] = 0
        #self._win_recv_count.Fence(MPI.MODE_NOSTORE)
        self._win_recv_count.Fence(0)
        for px in range(lcount):
            rk = lrank[px]
            self._win_recv_count.Get_accumulate(np.array(1), lrind[px:px+1], rk)
        self._win_recv_count.Fence(MPI.MODE_NOSTORE)
        
        nbytes = self._get_nbytes()

        send = np.zeros((lcount, nbytes), np.byte)
        for px in range(lcount):
            s = 0
            for dat in self.state.particle_dats:
                w = self._byte_per_element(dat)
                n = self._dat_ncomp(dat)
                w *= n
                v = send[px, s:s+w:].view(self._dat_dtype(dat))
                s += w
                v[:] = self._dat_obj(dat).view[px, :].copy()

        # need to place the data in the remote buffers here
        self._check_recv_win()

        # RMA 
        self._win_recv.Fence(0)

        for px in range(lcount):
            self._win_recv.Put(send[px, :], lrank[px], lrind[px])

        self._win_recv.Fence(MPI.MODE_NOSTORE)
        
        # unpack the data recv'd into dats
        old_npart_local = self.state.npart_local
        self.state.npart_local = old_npart_local + self._recv_count[0]

        for px in range(self._recv_count[0]):
            s = 0
            for dat in self.state.particle_dats:
                w = self._byte_per_element(dat)
                n = self._dat_ncomp(dat)
                w *= n
                v = self._recv[px, s:s+w:].view(self._dat_dtype(dat))
                s += w
                self._dat_obj(dat).view[old_npart_local + px, :] = v[:]
        
        self.state.remove_by_slot(lpid[:lcount])

















        




