from __future__ import print_function, division, absolute_import

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

# system level
import ctypes
import numpy as np
import time

# package level
from ppmd.lib import build
from ppmd import mpi, opt

REAL = ctypes.c_double
INT64 = ctypes.c_int64


MPI = mpi.MPI
AllocMem = mpi.AllocMem


class ParticleDatModifier:

    def __init__(self, dat, is_positiondat):
        self.dat = dat
        self.is_positiondat = is_positiondat

    def __enter__(self):
        return self.dat.view

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dat.sync_view_to_data()
        self.dat.mark_halos_old()
        if self.is_positiondat:
            self.dat.group.invalidate_lists = True
            # need to trigger MPI rank consistency/ownership here
            self.dat.group.check_position_consistency()


class GlobalDataMover:

    def __init__(self, state):
        self.state = state
        self.comm = state.domain.comm

        self._win_recv_count = None

        self._recv = None

        self._send = None
        self._win_recv = None

        self._recv_count = AllocMem(shape=(1,), dtype=INT64)
        

        self._key_call = self.__class__.__name__ + ':__call__'
        self._key_call_count = self.__class__.__name__ + ':__call__:count'
        self._key_check = self.__class__.__name__ + ':win_check'
        self._key_rma1 = self.__class__.__name__ + ':RMA_Get_accumulate'
        self._key_rma2 = self.__class__.__name__ + ':RMA_Put'
        self._key_local = self.__class__.__name__ + ':local_movement'
        self._key_compress = self.__class__.__name__ + ':compress'
        self._key_nsend = self.__class__.__name__ + ':nsend'
        self._key_nrecv = self.__class__.__name__ + ':nrecv'

        opt.PROFILE[self._key_call] = 0.0
        opt.PROFILE[self._key_call_count] = 0
        opt.PROFILE[self._key_check] = 0.0
        opt.PROFILE[self._key_rma1] = 0.0
        opt.PROFILE[self._key_rma2] = 0.0
        opt.PROFILE[self._key_local] = 0.0
        opt.PROFILE[self._key_compress] = 0.0
        opt.PROFILE[self._key_nsend] = 0
        opt.PROFILE[self._key_nrecv] = 0


    def _check_send_buffer(self, lcount, nbytes):
        
        if self._send is None or \
                self._send.shape[0] < lcount or \
                self._send.shape[1] != nbytes:

            self._send = np.zeros((lcount, nbytes), np.byte)

    
    def _check_recv_count_win(self):

        self._recv_count.array[0] = 0
        assert self._win_recv_count is None
        self._win_recv_count = MPI.Win.Create(self._recv_count.array, comm=self.comm)


    def _check_recv_win(self):
        assert self._win_recv is None
        t0 = time.time()
        nbytes = self._get_nbytes()

        # MPI Win create calls are collective on the comm
        if (self._recv is None) or \
                (self._recv.array.shape[0] < self._recv_count.array[0]) or \
                (self._recv.array.shape[1] != nbytes):

            if self._recv is not None:
                del self._recv

            nrow = self._recv_count.array[0]+100
            self._recv = AllocMem(shape=(nrow, nbytes), dtype=ctypes.c_char)

        self._win_recv = MPI.Win.Create(self._recv.array, disp_unit=nbytes, comm=self.comm)
        
        opt.PROFILE[self._key_check] += time.time() - t0

    
    def _free_wins(self):

        self._win_recv.Free()
        self._win_recv_count.Free()
        self._win_recv = None
        self._win_recv_count = None

    
    def __del__(self):
        del self._recv_count
        if self._recv is not None:
            del self._recv


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
        t0 = time.time()

        state = self.state
        comm = self.comm

        if comm.size == 1:
            return
        

        rank = comm.rank
        topo = mpi.cartcomm_top_xyz(comm)
        dims = mpi.cartcomm_dims_xyz(comm)

        extent = state.domain.extent
        boundary = state.domain.boundary

        dist_cell_widths = [1.0 / (ex / dx) for ex, dx in zip(extent, dims)]
        dist_cell_widths = np.array(dist_cell_widths, dtype=REAL)
        npart = state.npart_local
        pos = state.get_position_dat()
        pos = pos.view

        lcount = 0
        lrank_dict = {}
        lpid = []
        
        rk_offsets = (1, dims[0], dims[0]*dims[1])


        def to_mpi_rank(_p):
            # avoid send if possible
            if ((_p[0] >= boundary[0]) and (_p[0] <= boundary[1]) and \
                (_p[1] >= boundary[2]) and (_p[1] <= boundary[3]) and \
                (_p[2] >= boundary[4]) and (_p[2] <= boundary[5])):
                return rank
            
            # case where particle needs sending to another rank
            _rk = 0
            for dx in range(3):
                assert _p[dx] <=  0.5 * extent[dx], "outside domain"
                assert _p[dx] >= -0.5 * extent[dx], "outside domain"
                tint = int((_p[dx] + 0.5 * extent[dx]) * dist_cell_widths[dx])
                tint = min(dims[dx]-1, tint)
                _rk += tint * rk_offsets[dx]

            return _rk
        

        # find the new remote rank for leaving particles
        t0_local = time.time()
        for px in range(npart):
            rk = to_mpi_rank(pos[px])
            if rk != rank:
                lcount += 1
                if rk not in lrank_dict.keys():
                    lrank_dict[rk] = [px]
                else:
                    lrank_dict[rk].append(px)
                lpid.append(px)
        t_local = time.time() - t0_local
        num_rranks = len(lrank_dict.keys())
        
        # for each remote rank get accumalate
        t1 = time.time()
        self._check_recv_count_win()
        
        # prevent sizes going out of scope
        _size_store = []
        #self._win_recv_count.Fence(0)
        lrind = np.zeros((num_rranks, 2), INT64)
        

        for rki, rk in enumerate(lrank_dict.keys()):
            lrind[rki, 0] = rk
            _size = np.array((len(lrank_dict[rk]),), INT64)
            _size_store.append(_size)
            self._win_recv_count.Lock(rk, MPI.LOCK_SHARED)
            self._win_recv_count.Get_accumulate(_size_store[-1], lrind[rki, 1:2], rk)
            self._win_recv_count.Unlock(rk)
        

        self.comm.Barrier()
        #self._win_recv_count.Fence(MPI.MODE_NOSTORE)
        del _size_store

        opt.PROFILE[self._key_rma1] += time.time() - t1
        opt.PROFILE[self._key_nsend] += lcount
        
        # pack the send buffer for all particles
        t0_local = time.time()

        # get the views before the copy to avoid excessive syncing with the case
        # of CUDA
        views = {}
        bytes_per_element = {}
        for dat in self.state.particle_dats:
            views[dat] = self._dat_obj(dat).view.view()
            bytes_per_element[dat] = self._byte_per_element(dat)

        nbytes = self._get_nbytes()
        self._check_send_buffer(lcount, nbytes)
        send_offset = 0
        for rk in lrind[:, 0]:
            for px in lrank_dict[rk]:
                s = 0
                for dat in self.state.particle_dats:
                    w = bytes_per_element[dat]
                    n = self._dat_ncomp(dat)
                    w *= n
                    v = self._send[send_offset, s:s+w:].view(self._dat_dtype(dat))
                    s += w
                    v[:] = views[dat][px, :].copy()
                send_offset += 1
        t_local += time.time() - t0_local
        

        # need to place the data in the remote buffers here
        self._check_recv_win()
        
        t2 = time.time()
        # RMA 
        
        send_offset = 0
        for rki in range(num_rranks):
            rk = lrind[rki, 0]
            ri = lrind[rki, 1]
            nsend = len(lrank_dict[rk])
            self._win_recv.Lock(rk, MPI.LOCK_SHARED)
            #self._win_recv.Lock(rk, MPI.LOCK_EXCLUSIVE)
            self._win_recv.Put(
                self._send[send_offset:send_offset + nsend:, :],
                rk,
                ri
            )
            self._win_recv.Unlock(rk)
            send_offset += nsend
        
        self.comm.Barrier()
        opt.PROFILE[self._key_rma2] += time.time() - t2
        opt.PROFILE[self._key_nrecv] += self._recv_count.array[0]


        # unpack the data recv'd into dats
        old_npart_local = self.state.npart_local
        self.state.npart_local = old_npart_local + self._recv_count.array[0]
        
        # get the views before the copy to avoid excessive syncing with the case
        # of CUDA
        views = {}
        bytes_per_element = {}
        for dat in self.state.particle_dats:
            views[dat] = self._dat_obj(dat).view.view()
            bytes_per_element[dat] = self._byte_per_element(dat)

        t0_local = time.time()
        for px in range(self._recv_count.array[0]):
            s = 0
            for dat in self.state.particle_dats:
                w = bytes_per_element[dat]
                n = self._dat_ncomp(dat)
                w *= n
                v = self._recv.array[px, s:s+w:].view(self._dat_dtype(dat))
                s += w
                views[dat][old_npart_local + px, :] = v[:]
        
        
        # on some architectures the memory used for compute is different to the
        # exposed numpy view
        for dat in self.state.particle_dats:
            self._dat_obj(dat).sync_view_to_data()
        

        t_local += time.time() - t0_local
        opt.PROFILE[self._key_local] += t_local
        
        t0_compress = time.time()
        self.state.remove_by_slot(lpid)
        opt.PROFILE[self._key_compress] += time.time() - t0_compress
        
        self._free_wins()
        opt.PROFILE[self._key_call] += time.time() - t0
        opt.PROFILE[self._key_call_count] += 1








