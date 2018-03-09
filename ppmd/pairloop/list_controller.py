"""
Manage the rebuilding of neighbour lists and cell lists
"""

from ppmd.access import READ
from ppmd.pairloop.neighbour_matrix_omp_sub import NeighbourListOMPSub
from ppmd.plain_cell_list import PlainCellList


class NListController(object):
    """
    Manage neighbour list rebuilding
    """
    def __init__(self):
        self._dict = {}

    def get_neighbour_list(self, position_dat, cutoff):
        key = (position_dat, cutoff)
        if not key in self._dict.keys():
            self._create_list(key)

        self._update_list(key)
        return self._dict[key]['nlist']
    
    def _update_list(self, key):
        posdat = key[0]
        width = key[1]       
        it = self._dict[key]
        g = posdat.group
        
        if g.domain.version_id > it['domain_id']:
            # if the domain has changed rebuild 
            del self._dict[key]
            self._create_list(key)
            it = self._dict[key]
            
        # ensure the cell to particle map for the group is up-to-date
        g.get_cell_to_particle_map().check()
        
        # neighbour list update requires up-to-date halos
        posdat.ctypes_data_access(READ, pair=True)

        # build the subcell list and create the neighbour list
        gvid = g.get_cell_to_particle_map().version_id
        cl = it['clist']
        if cl.version_id < gvid:
            cl.sort(posdat, g.npart_local + g.npart_halo)
            cl.version_id = gvid
            it['nlist'].update(g.npart_local, posdat)
    
    def _create_list(self, key):
        posdat = key[0]
        width = key[1]
        
        if posdat.group is None:
            raise RuntimeError('position dat has no group and hence no domain')
        
        g = posdat.group
        
        decomp_flag = g.cell_decompose(width)
        if decomp_flag:
            # all existing cell lists and neighbour lists
            # are now invalid
            self._dict = {}

        cl = PlainCellList(width, g.domain.boundary,
                n=g.npart_local+g.npart_halo)
        nl = NeighbourListOMPSub(width, cl, n=g.npart_local)
        
        self._dict[key] = {}
        self._dict[key]['clist'] = cl
        self._dict[key]['nlist'] = nl
        self._dict[key]['domain_id'] = g.domain.version_id


nlist_controller = NListController()
"""
Global neighbour list controller
"""


