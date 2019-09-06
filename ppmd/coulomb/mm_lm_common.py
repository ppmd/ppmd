from __future__ import division, print_function, absolute_import

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


import numpy as np

from ppmd import data, loop, kernel, access, lib, opt, pairloop
from ppmd.coulomb import fmm_interaction_lists

from ppmd.coulomb.sph_harm import *

from ppmd.coulomb.fmm_pbc import LongRangeMTL


import ctypes
import math
import time

Constant = kernel.Constant
REAL = ctypes.c_double
INT64 = ctypes.c_int64

PROFILE = opt.PROFILE


from enum import Enum
class BCType(Enum):
    """
    Enum to indicate boundary condition type.
    """
    PBC = 'pbc'
    """Fully periodic boundary conditions"""
    FREE_SPACE = 'free_space'
    """Free-space, e.g. vacuum, boundary conditions."""
    NEAREST = '27'
    """Primary image and the surrounding 26 nearest neighbours."""



class MM_LM_Common:

    def __init__(self, positions, charges, domain, boundary_condition, r, l):

        self.positions = positions
        self.charges = charges
        self.domain = domain
        self.comm = self.domain.comm
        self.boundary_condition = BCType(boundary_condition)
        self.R = r
        self.L = l
        self.ncomp = (self.L ** 2) * 2
        self.group = self.positions.group
        self.subdivision = (2, 2, 2)

        # interaction lists
        self.il = fmm_interaction_lists.compute_interaction_lists(domain.extent, self.subdivision)
        self.il_max_len = max(len(lx) for lx in self.il[0])
        self.il_array = np.array(self.il[0], INT64)
        self.il_scalararray = data.ScalarArray(ncomp=self.il_array.size, dtype=INT64)
        self.il_scalararray[:] = self.il_array.ravel().copy()
        self.il_earray = np.array(self.il[1], INT64)
        
        # tree is stored in a GlobalArray
        n = 0
        for rx in range(self.R):
            s = [sx ** rx for sx in self.subdivision]
            n += s[0] * s[1] * s[2] * self.ncomp
        self.tree = data.GlobalArray(ncomp=n, dtype=REAL)

        self._init_dats()

        s = self.subdivision
        s = [sx ** (self.R - 1) for sx in s]
        ncells_finest = s[0] * s[1] * s[2]
        self.cell_occupation_ga = data.GlobalArray(ncomp=ncells_finest, dtype=INT64)


        self.sph_gen = SphGen(l-1)

        self.widths_x = data.ScalarArray(ncomp=self.R, dtype=REAL)
        self.widths_y = data.ScalarArray(ncomp=self.R, dtype=REAL)
        self.widths_z = data.ScalarArray(ncomp=self.R, dtype=REAL)
        
        e = self.domain.extent
        s = self.subdivision
        self.widths_x[:] = [e[0] / (s[0] ** rx) for rx in range(self.R)]
        self.widths_y[:] = [e[1] / (s[1] ** rx) for rx in range(self.R)]
        self.widths_z[:] = [e[2] / (s[2] ** rx) for rx in range(self.R)]

        self.ncells_x = data.ScalarArray(ncomp=self.R, dtype=INT64)
        self.ncells_y = data.ScalarArray(ncomp=self.R, dtype=INT64)
        self.ncells_z = data.ScalarArray(ncomp=self.R, dtype=INT64)
        self.ncells_x[:] = [int(s[0] ** rx) for rx in range(self.R)]
        self.ncells_y[:] = [int(s[1] ** rx) for rx in range(self.R)]
        self.ncells_z[:] = [int(s[2] ** rx) for rx in range(self.R)]

        self._extract_energy = data.GlobalArray(ncomp=1, dtype=REAL)

        
        # find the max distance in the direct part using the exclusion lists
        widths = [ex / (sx ** (self.R - 1)) for ex, sx in zip(self.domain.extent, self.subdivision)]
        max_cell_width = max(
            [(abs(ix[0]) + 1) * widths[0] for ix in self.il[1]] + \
            [(abs(ix[1]) + 1) * widths[1] for ix in self.il[1]] + \
            [(abs(ix[2]) + 1) * widths[2] for ix in self.il[1]]
        )
        self.max_cell_width = max_cell_width

        self.widths_x_str = ','.join([str(ix) for ix in  self.widths_x])
        self.widths_y_str = ','.join([str(ix) for ix in  self.widths_y])
        self.widths_z_str = ','.join([str(ix) for ix in  self.widths_z])
        self.ncells_x_str = ','.join([str(ix) for ix in  self.ncells_x])
        self.ncells_y_str = ','.join([str(ix) for ix in  self.ncells_y])
        self.ncells_z_str = ','.join([str(ix) for ix in  self.ncells_z])
        
        level_offsets = [0]
        nx = 1
        ny = 1
        nz = 1
        for level in range(1, self.R):
            level_offsets.append(
                level_offsets[-1] + nx * ny * nz * self.ncomp
            )
            nx *= self.subdivision[0]
            ny *= self.subdivision[1]
            nz *= self.subdivision[2]

        self.level_offsets_str = ','.join([str(ix) for ix in level_offsets])


        self._contrib_loop = None
        self._init_contrib_loop()

        self._extract_loop = None
        self._init_extract_loop()


        self.sh = pairloop.state_handler.StateHandler(state=None, shell_cutoff=max_cell_width)
        
        self.cell_list = np.zeros(1000, INT64)
        self.cell_occ = np.zeros(1000, INT64)
        self.cell_remaps = np.zeros((1000, 3), INT64)

        self._direct_lib = None
        self._cell_remap_lib = None
        self._init_direct_libs()

        if self.boundary_condition == BCType.PBC:
            self._init_pbc()



    def _init_direct_libs(self):

        
        bc = self.boundary_condition
        if bc == BCType.FREE_SPACE:
            bc_block = r'''
                // free space BCs
                if (ocx < 0) {{continue;}}
                if (ocy < 0) {{continue;}}
                if (ocz < 0) {{continue;}}
                if (ocx >= LCX) {{continue;}}
                if (ocy >= LCY) {{continue;}}
                if (ocz >= LCZ) {{continue;}}
            '''
        elif bc in (BCType.NEAREST, BCType.PBC):
            bc_block = ''
        else:
            raise RuntimeError('Unknown boundary condition.')

        source = r'''
        extern "C" int direct_interactions(
            const INT64            N_TOTAL,
            const INT64            N_LOCAL,
            const INT64            MAX_OCC, 
            const INT64            NOFFSETS,                //number of nearest neighbour cells
            const INT64 * RESTRICT NNMAP,                   //nearest neighbour offsets
            const REAL  * RESTRICT positions,
            const REAL  * RESTRICT charges,
            const INT64 * RESTRICT cells,
            const INT64 * RESTRICT cell_mins,
            const INT64 * RESTRICT cell_counts,
                  INT64 * RESTRICT cell_occ,
                  INT64 * RESTRICT cell_list,
                  REAL  * RESTRICT total_energy
        ){{ 

            // bin into cells
            for(INT64 ix=0 ; ix<N_TOTAL ; ix++){{

                const INT64 cx = cells[ix*3 + 0] - cell_mins[0];
                const INT64 cy = cells[ix*3 + 1] - cell_mins[1];
                const INT64 cz = cells[ix*3 + 2] - cell_mins[2];
                const INT64 clin = cx + cell_counts[0] * (cy + cell_counts[1] * cz);
                INT64 layer = -1;
                
                layer = cell_occ[clin]++;

                cell_list[clin * MAX_OCC + layer] = ix;
            }}


            // direct interactions for local charges
            REAL outer_energy = 0.0;
            #pragma omp parallel for reduction(+:outer_energy)
            for(INT64 ix=0 ; ix<N_LOCAL ; ix++){{

                const REAL pix = positions[ix * 3 + 0];
                const REAL piy = positions[ix * 3 + 1];
                const REAL piz = positions[ix * 3 + 2];
                const INT64 cix = cells[ix * 3 + 0];
                const INT64 ciy = cells[ix * 3 + 1];
                const INT64 ciz = cells[ix * 3 + 2];
                const REAL qi = charges[ix];
                
                REAL tmp_energy = 0.0;
                for(INT64 ox=0 ; ox<NOFFSETS ; ox++){{

                    INT64 ocx = cix + NNMAP[ox * 3 + 0];
                    INT64 ocy = ciy + NNMAP[ox * 3 + 1];
                    INT64 ocz = ciz + NNMAP[ox * 3 + 2];
                    
                    {BC_BLOCK}

                    ocx -= cell_mins[0];
                    ocy -= cell_mins[1];
                    ocz -= cell_mins[2];

                    // if a plane of edge cells is empty they may not exist in the data structure
                    if (ocx < 0) {{continue;}}
                    if (ocy < 0) {{continue;}}
                    if (ocz < 0) {{continue;}}
                    if (ocx >= cell_counts[0]) {{continue;}}
                    if (ocy >= cell_counts[1]) {{continue;}}
                    if (ocz >= cell_counts[2]) {{continue;}}                   

                    const INT64 cj = ocx + cell_counts[0] * (ocy + cell_counts[1] * ocz);

                    const int mask = (NNMAP[ox * 3 + 0] == 0) && (NNMAP[ox * 3 + 1] == 0) && (NNMAP[ox * 3 + 2] == 0);
                    
                    if (mask) {{
                        for(INT64 jxi=0 ; jxi<cell_occ[cj] ; jxi++){{
                            
                            const INT64 jx = cell_list[cj * MAX_OCC + jxi];

                            const REAL pjx = positions[jx * 3 + 0];
                            const REAL pjy = positions[jx * 3 + 1];
                            const REAL pjz = positions[jx * 3 + 2];
                            const REAL qj = charges[jx];


                            const REAL dx = pix - pjx;
                            const REAL dy = piy - pjy;
                            const REAL dz = piz - pjz;

                            const REAL r2 = dx*dx + dy*dy + dz*dz;
                            
                            tmp_energy += (ix == jx) ? 0.0 :  qj / sqrt(r2);
                        

                        }}

                    }} else {{
                        for(INT64 jxi=0 ; jxi<cell_occ[cj] ; jxi++){{
                            
                            const INT64 jx = cell_list[cj * MAX_OCC + jxi];

                            const REAL pjx = positions[jx * 3 + 0];
                            const REAL pjy = positions[jx * 3 + 1];
                            const REAL pjz = positions[jx * 3 + 2];
                            const REAL qj = charges[jx];


                            const REAL dx = pix - pjx;
                            const REAL dy = piy - pjy;
                            const REAL dz = piz - pjz;

                            const REAL r2 = dx*dx + dy*dy + dz*dz;
                            
                            tmp_energy += qj / sqrt(r2);

                        }}
                    }}

                }}



                outer_energy += tmp_energy * 0.5 * qi;
            }}

            total_energy[0] = outer_energy;

            return 0;
        }}

        '''.format(
            BC_BLOCK=bc_block
        )

        header = r'''
        #include <math.h>
        #include <stdio.h>
        #define REAL double
        #define INT64 int64_t
        #define LCX {LCX}
        #define LCY {LCY}
        #define LCZ {LCZ}
        '''.format(
            LCX=self.subdivision[0] ** (self.R - 1),
            LCY=self.subdivision[1] ** (self.R - 1),
            LCZ=self.subdivision[2] ** (self.R - 1)
        )

        self._direct_lib = lib.build.simple_lib_creator(header, source)['direct_interactions']
    


        source = r'''
        extern "C" int cell_remap(
            const INT64            N_TOTAL,
            const INT64            N_LOCAL,
            const REAL  * RESTRICT positions,
            const INT64 * RESTRICT cells,
                  INT64 * RESTRICT cell_remaps
        ){{ 

            // bin into cells
            #pragma omp parallel for
            for(INT64 ix=0 ; ix<N_TOTAL ; ix++){{

                const INT64 ocx = cells[ix * 3 + 0];
                const INT64 ocy = cells[ix * 3 + 1];
                const INT64 ocz = cells[ix * 3 + 2];

                if(ix < N_LOCAL){{

                    cell_remaps[ix * 3 + 0] = ocx;
                    cell_remaps[ix * 3 + 1] = ocy;
                    cell_remaps[ix * 3 + 2] = ocz;

                }} else {{
                    
                    INT64 tcx = ocx;
                    INT64 tcy = ocy;
                    INT64 tcz = ocz;

                    const REAL hpx = positions[3*ix + 0];
                    const REAL hpy = positions[3*ix + 1];
                    const REAL hpz = positions[3*ix + 2];
                    

                    if ((hpx >= HEX) && (ocx < (LCX-1)))    {{ tcx += LCX; }}
                    if ((hpx <= -1.0*HEX) && (  ocx > 0 ))  {{ tcx -= LCX; }}
                    if ((hpy >= HEY) && ( ocy < (LCX-1) ))  {{ tcy += LCY; }}
                    if ((hpy <= -1.0*HEY) && ( ocy > 0 ) )  {{ tcy -= LCY; }}
                    if ((hpz >= HEZ) && (ocz < (LCX-1)))    {{ tcz += LCZ; }}
                    if ((hpz <= -1.0*HEZ) && (ocz > 0))     {{ tcz -= LCZ; }}


                    cell_remaps[ix * 3 + 0] = tcx;
                    cell_remaps[ix * 3 + 1] = tcy;
                    cell_remaps[ix * 3 + 2] = tcz;

                }}

            }}

            return 0;
        }}

        '''.format(
        )

        header = r'''
        #include <math.h>
        #include <stdio.h>
        #define REAL double
        #define INT64 int64_t
        #define LCX {LCX}
        #define LCY {LCY}
        #define LCZ {LCZ}
        #define EX {EX}
        #define EY {EY}
        #define EZ {EZ}
        #define HEX {HEX}
        #define HEY {HEY}
        #define HEZ {HEZ}        
        '''.format(
            LCX=self.subdivision[0] ** (self.R - 1),
            LCY=self.subdivision[1] ** (self.R - 1),
            LCZ=self.subdivision[2] ** (self.R - 1),
            EX=self.domain.extent[0],
            EY=self.domain.extent[1],
            EZ=self.domain.extent[2],
            HEX=self.domain.extent[0] * 0.5,
            HEY=self.domain.extent[1] * 0.5,
            HEZ=self.domain.extent[2] * 0.5            
        )

        self._cell_remap_lib = lib.build.simple_lib_creator(header, source)['cell_remap']






    def _execute_direct(self, positions, charges, forces=None, potential=None):

        t0 = time.time()

        dats = {
                'P': positions(access.READ),
                'Q': charges(access.READ),
                'C': self._dat_fine_cells(access.READ)
        }

        nlocal, nhalo, ncell = self.sh.pre_execute(dats=dats)

        if self.cell_remaps.shape[0] < (nlocal + nhalo):
            self.cell_remaps = np.zeros((nlocal + nhalo + 100, 3), INT64)

        self._cell_remap_lib(
            INT64(nlocal + nhalo),
            INT64(nlocal),
            self.sh.get_pointer(positions(access.READ)),
            self.sh.get_pointer(self._dat_fine_cells(access.READ)),
            self.cell_remaps.ctypes.get_as_parameter()
        )

        minc = np.array(np.min(self.cell_remaps[:nlocal + nhalo, :], 0), dtype=INT64)
        self.minc = minc
        maxc = np.array(np.max(self.cell_remaps[:nlocal + nhalo, :], 0), dtype=INT64)
        max_occ = np.max(self.cell_occupation_ga[:])
        self.max_occ = max_occ
        widths = np.array(maxc - minc + 1, INT64) 
        self.widths = widths

        ncells = widths[0] * widths[1] * widths[2]
        
        # storage for cell to particle map
        if self.cell_occ.shape[0] < ncells:
            self.cell_occ = np.zeros(ncells + 100, INT64)
        if self.cell_list.shape[0] < (ncells * max_occ):
            self.cell_list = np.zeros(ncells * max_occ + 100, INT64)

        self.cell_occ[:] = 0
        
        tmp_energy = np.zeros(1, REAL)

        t0d = time.time()
        self._direct_lib(
            INT64(nlocal + nhalo),
            INT64(nlocal),
            INT64(max_occ), 
            INT64(self.il_earray.shape[0]),
            self.il_earray.ctypes.get_as_parameter(),
            self.sh.get_pointer(positions(access.READ)),
            self.sh.get_pointer(charges(access.READ)),
            self.cell_remaps.ctypes.get_as_parameter(),
            minc.ctypes.get_as_parameter(),
            widths.ctypes.get_as_parameter(),
            self.cell_occ.ctypes.get_as_parameter(),
            self.cell_list.ctypes.get_as_parameter(),
            tmp_energy.ctypes.get_as_parameter()
        )
        t1d = time.time()
        
        recv_energy = np.zeros_like(tmp_energy)
        self.domain.comm.Allreduce(tmp_energy, recv_energy)

        self.sh.post_execute(dats=dats)

        t1 = time.time()
        
        self._profile_inc('direct_python', t1 - t0)
        self._profile_inc('direct_c', t1d - t0d)

        return recv_energy[0]



    def _profile_inc(self, key, inc):
        key = self.__class__.__name__ + ':' + key
        if key not in PROFILE.keys():
            PROFILE[key] = inc
        else:
            PROFILE[key] += inc



    def _init_pbc(self):
        
        self.top_multipole_expansion_ga = data.GlobalArray(ncomp=self.ncomp, dtype=REAL)
        self.top_dot_vector_ga = data.GlobalArray(ncomp=self.ncomp, dtype=REAL)
        self.lrc = LongRangeMTL(self.L, self.domain, exclude_tuples=self.il[1])


        sph_gen = self.sph_gen

        def cube_ind(L, M):
            return ((L) * ( (L) + 1 ) + (M) )

        assign_gen =  'double rhol = charge;\n'
        for lx in range(self.L):
            for mx in range(-lx, lx+1):

                res, ims = sph_gen.get_y_sym(lx, -mx)
                offset = cube_ind(lx, mx)

                assign_gen += ''.join(['MULTIPOLE[{}] += {} * rhol;\n'.format(*args) for args in (
                        (offset, str(res)),
                        (offset + self.L**2, str(ims))
                    )
                ])

                res, ims = sph_gen.get_y_sym(lx, mx)
                assign_gen += ''.join(['DOT_VEC[{}] += {} * rhol;\n'.format(*args) for args in (
                        (offset, str(res)),
                        (offset + self.L**2, '-1.0 * ' + str(ims))
                    )
                ])

            assign_gen += 'rhol *= radius;\n'


        lr_kernel = kernel.Kernel(
            'mm_lm_lr_kernel',
            r'''
            const double dx = P.i[0];
            const double dy = P.i[1];
            const double dz = P.i[2];

            const double xy2 = dx * dx + dy * dy;
            const double radius = sqrt(xy2 + dz * dz);
            const double theta = atan2(sqrt(xy2), dz);
            const double phi = atan2(dy, dx);
            const double charge = Q.i[0];

            {SPH_GEN}
            {ASSIGN_GEN}


            '''.format(
                SPH_GEN=str(sph_gen.module),
                ASSIGN_GEN=str(assign_gen)
            ),
            headers=(
                lib.build.write_header(
                    r'''
                    #include <math.h>
                    '''
                ),
            )
        )

        
        self._lr_loop = loop.ParticleLoopOMP(
            lr_kernel,
            dat_dict={
                'P': self.positions(access.READ),
                'Q': self.charges(access.READ),
                'MULTIPOLE': self.top_multipole_expansion_ga(access.INC_ZERO),
                'DOT_VEC': self.top_dot_vector_ga(access.INC_ZERO),
            }
        )


    def _compute_pbc_contrib(self):

        if not self.boundary_condition == BCType.PBC:
            return 0.0
        
        self._lr_loop.execute()

        multipole_exp = self.top_multipole_expansion_ga[:].copy()
        L_tmp = np.zeros_like(multipole_exp)

        self.lrc(multipole_exp, L_tmp)

        self.mvector = multipole_exp
        self.evector = self.top_dot_vector_ga[:].copy()

        self.lr_energy = 0.5 * np.dot(self.evector, L_tmp)


        return self.lr_energy




















