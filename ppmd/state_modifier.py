__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


import ctypes
import numpy as np

INT64 = ctypes.c_int64


class StateModifierContext:

    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self.state.modifier

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.state.modifier.check_consistency()



class StateModifier:

    def __init__(self, state):
        self.state = state
        
        # initial reset
        self._to_add = []
        self._to_remove = set()

    def _reset(self):
        self._to_add = []
        self._to_remove = set()

    def add(self, values):
        if len(values) > 0:
            self._to_add.append(values)

    def remove(self, indices):
        indices = set(indices)
        for ix in indices:
            assert ix >= 0
            assert ix < self.state.npart_local

        self._to_remove = self._to_remove.union(set(indices))

    def check_consistency(self):
        
        num_new_particles = 0
        for dictx in self._to_add:
            num_new_particles += np.atleast_2d(next(iter(dictx.values()))).shape[0]
        
        old_npart_local = self.state.npart_local
        new_npart_local = old_npart_local + num_new_particles
        
        # set new npart_local (also resizes dats if needed)
        self.state.npart_local = new_npart_local

        position_dat = self.state.get_position_dat()
        
        # loop over dats and copy in the new data
        for datx in self.state.particle_dats:
            dat = getattr(self.state, datx)
            
            # need to do positions last
            if dat == position_dat: continue
            
            with dat.modify_view() as m:
                
                # copy in new data
                particle_index = old_npart_local
                for dictx in self._to_add:

                    # get the size for indexing
                    bs = np.atleast_2d(next(iter(dictx.values()))).shape[0]
                    te = particle_index + bs
                    
                    # copy the new values if they exist or zero the non passed entries
                    if dat in dictx.keys():
                        b = np.atleast_2d(dictx[dat])
                        m[particle_index:te:, :] = b.copy()
                    else:
                        m[particle_index:te:, :].fill(0)

                    particle_index = te


        # do the positions last to trigger domain decomp consistency after the other
        # values are in place in the dats

        _t = np.array((0,), dtype=INT64)
        _o = np.array((new_npart_local,), dtype=INT64)
        self.state.domain.comm.Allreduce(_o, _t)
        npart = _t[0]
        self.state.npart = npart


        # copy in new data
        particle_index = old_npart_local

        with position_dat.modify_view() as m:
            for dictx in self._to_add:

                # get the size for indexing
                bs = np.atleast_2d(next(iter(dictx.values()))).shape[0]

                te = particle_index + bs
                
                # copy the new values if they exist or zero the non passed entries
                if position_dat in dictx.keys():
                    b = np.atleast_2d(dictx[position_dat])
                    m[particle_index:te:, :] = b.copy()
                else:
                    m[particle_index:te:, :].fill(0)

                particle_index = te

        # remove the removed particles
        
        # the position modifier will potentially set a new npart_local on the state
        npart_local = self.state.npart_local - len(self._to_remove)

        _t = np.array((0,), dtype=INT64)
        _o = np.array((npart_local,), dtype=INT64)
        self.state.domain.comm.Allreduce(_o, _t)
        npart = _t[0]
        self.state.npart = npart

        self.state.remove_by_slot(tuple(self._to_remove))
        self._reset()









