"""
Multipole expansion based long range electrostatics O(NlogN).
"""
from __future__ import division, print_function, absolute_import

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


import numpy as np
from ppmd import data, loop, kernel, access, lib, opt, pairloop
from ppmd.coulomb import fmm_interaction_lists

from ppmd.coulomb.sph_harm import *

import ctypes
import math
import time

Constant = kernel.Constant
REAL = ctypes.c_double
INT64 = ctypes.c_int64

PROFILE = opt.PROFILE

class PyMM:

    def __init__(self, positions, charges, domain, boundary_condition, r, l):

        assert boundary_condition == 'free_space'

        self.positions = positions
        self.charges = charges
        self.domain = domain
        self.comm = self.domain.comm
        self.boundary_condition = boundary_condition
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

        g = self.group
        pd = type(self.charges)
        # xyz fmm cells
        g._mm_fine_cells = pd(ncomp=3, dtype=INT64)
        g._mm_cells = pd(ncomp=3*self.R, dtype=INT64)
        g._mm_child_index = pd(ncomp=3*self.R, dtype=INT64)

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


        self._contrib_loop = None
        self._init_contrib_loop()

        self._extract_loop = None
        self._init_extract_loop()

        # direct part
        max_cell_width = max((self.widths_x[-1], self.widths_y[-1], self.widths_z[-1])) * 2.0
        
        # find the max distance in the direct part using the exclusion lists
        widths = [ex / (sx ** (self.R - 1)) for ex, sx in zip(self.domain.extent, self.subdivision)]
        max_cell_width = max(
            [(abs(ix[0]) + 1) * widths[0] for ix in self.il[1]] + \
            [(abs(ix[1]) + 1) * widths[1] for ix in self.il[1]] + \
            [(abs(ix[2]) + 1) * widths[2] for ix in self.il[1]]
        )
        self.sh = pairloop.state_handler.StateHandler(state=None, shell_cutoff=max_cell_width)
        
        self.cell_list = np.zeros(1000, INT64)
        self.cell_occ = np.zeros(1000, INT64)
        self.cell_remaps = np.zeros((1000, 3), INT64)

        self._direct_lib = None
        self._cell_remap_lib = None
        self._init_direct_libs()


    def _init_direct_libs(self):


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
                    
                    // free space BCs
                    if (ocx < 0) {{continue;}}
                    if (ocy < 0) {{continue;}}
                    if (ocz < 0) {{continue;}}
                    if (ocx >= LCX) {{continue;}}
                    if (ocy >= LCY) {{continue;}}
                    if (ocz >= LCZ) {{continue;}}
                    
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
                'C': self.group._mm_fine_cells(access.READ)
        }

        nlocal, nhalo, ncell = self.sh.pre_execute(dats=dats)

        if self.cell_remaps.shape[0] < (nlocal + nhalo):
            self.cell_remaps = np.zeros((nlocal + nhalo + 100, 3), INT64)

        self._cell_remap_lib(
            INT64(nlocal + nhalo),
            INT64(nlocal),
            self.sh.get_pointer(positions(access.READ)),
            self.sh.get_pointer(self.group._mm_fine_cells(access.READ)),
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


    def _init_contrib_loop(self):
        
        g = self.group
        extent = self.domain.extent
        cell_widths = [1.0 / (ex / (sx**(self.R - 1))) for ex, sx in zip(extent, self.subdivision)]


        sph_gen = self.sph_gen

        def cube_ind(L, M):
            return ((L) * ( (L) + 1 ) + (M) )

        assign_gen =  'double rhol = 1.0;\n'
        assign_gen += 'double rholcharge = rhol * charge;\n'
        for lx in range(self.L):
            for mx in range(-lx, lx+1):
                assign_gen += 'TREE[OFFSET + {ind}] += {ylmm} * rholcharge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, -mx)[0])
                    )
                assign_gen += 'TREE[OFFSET + IM_OFFSET + {ind}] += {ylmm} * rholcharge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, -mx)[1])
                    )
            assign_gen += 'rhol *= radius;\n'
            assign_gen += 'rholcharge = rhol * charge;\n'



        k = kernel.Kernel(
            'mm_contrib_loop',
            r'''
            
            const double rx = P.i[0];
            const double ry = P.i[1];
            const double rz = P.i[2];

            // bin into finest level cell
            const double srx = rx + HEX;
            const double sry = ry + HEY;
            const double srz = rz + HEZ;
            
            int64_t cfx = srx * CWX;
            int64_t cfy = sry * CWY;
            int64_t cfz = srz * CWZ;

            cfx = (cfx < LCX) ? cfx : (LCX - 1);
            cfy = (cfy < LCX) ? cfy : (LCY - 1);
            cfz = (cfz < LCX) ? cfz : (LCZ - 1);
            
            // number of cells in each direction
            int64_t ncx = LCX;
            int64_t ncy = LCY;
            int64_t ncz = LCZ;
            
            // increment the occupancy for this cell
            OCC_GA[cfx + LCX * (cfy + LCY * cfz)]++;

            MM_FINE_CELLS.i[0] = cfx;
            MM_FINE_CELLS.i[1] = cfy;
            MM_FINE_CELLS.i[2] = cfz;

            for( int level=R-1 ; level>=0 ; level-- ){{
                
                // // cell widths for cell centre computation
                // const double wx = EX / ncx;
                // const double wy = EY / ncy;
                // const double wz = EZ / ncz;

                // child on this level

                const int64_t cix = cfx % SDX;
                const int64_t ciy = cfy % SDY;
                const int64_t ciz = cfz % SDZ;

                // record the cell indices
                MM_CELLS.i[level * 3 + 0] = cfx;
                MM_CELLS.i[level * 3 + 1] = cfy;
                MM_CELLS.i[level * 3 + 2] = cfz;

                // record the child cell indices
                MM_CHILD_INDEX.i[level * 3 + 0] = cix;
                MM_CHILD_INDEX.i[level * 3 + 1] = ciy;
                MM_CHILD_INDEX.i[level * 3 + 2] = ciz;

                // compute the cells for the next level
                cfx /= SDX;
                cfy /= SDY;
                cfz /= SDZ;

                //// number of cells in each dim for the next level
                //ncx /= SDX;
                //ncy /= SDY;
                //ncz /= SDZ;
            }}


            int64_t LEVEL_OFFSETS[R];
            LEVEL_OFFSETS[0] = 0;
            int64_t nx = 1;
            int64_t ny = 1;
            int64_t nz = 1;
            for(int level=1 ; level<R ; level++ ){{
                int64_t nprev = nx * ny * nz * NCOMP;
                LEVEL_OFFSETS[level] = LEVEL_OFFSETS[level - 1] + nprev;
                nx *= SDX;
                ny *= SDY;
                nz *= SDZ;
            }}


            // compute the multipole expansions
            for( int level=0 ; level<R ; level++) {{

                const int64_t cellx = MM_CELLS.i[level * 3 + 0];
                const int64_t celly = MM_CELLS.i[level * 3 + 1];
                const int64_t cellz = MM_CELLS.i[level * 3 + 2];

                const double dx = rx - ((-HEX) + (0.5  + cellx) * WIDTHS_X[level]);
                const double dy = ry - ((-HEY) + (0.5  + celly) * WIDTHS_Y[level]);
                const double dz = rz - ((-HEZ) + (0.5  + cellz) * WIDTHS_Z[level]);

                const double xy2 = dx * dx + dy * dy;
                const double radius = sqrt(xy2 + dz * dz);
                const double theta = atan2(sqrt(xy2), dz);
                const double phi = atan2(dy, dx);
                const double charge = Q.i[0];
                
                const int64_t lin_ind = cellx + NCELLS_X[level] * (celly + NCELLS_Y[level] * cellz);
                const int64_t OFFSET = LEVEL_OFFSETS[level] + NCOMP * lin_ind;

                {SPH_GEN}
                {ASSIGN_GEN}
            }}


            '''.format(
                SPH_GEN=str(sph_gen.module),
                ASSIGN_GEN=str(assign_gen)
            ),
            (
                Constant('R', self.R),
                Constant('EX', extent[0]),
                Constant('EY', extent[1]),
                Constant('EZ', extent[2]),
                Constant('HEX', 0.5 * extent[0]),
                Constant('HEY', 0.5 * extent[1]),
                Constant('HEZ', 0.5 * extent[2]),                
                Constant('CWX', cell_widths[0]),
                Constant('CWY', cell_widths[1]),
                Constant('CWZ', cell_widths[2]),
                Constant('LCX', self.subdivision[0] ** (self.R - 1)),
                Constant('LCY', self.subdivision[1] ** (self.R - 1)),
                Constant('LCZ', self.subdivision[2] ** (self.R - 1)),
                Constant('SDX', self.subdivision[0]),
                Constant('SDY', self.subdivision[1]),
                Constant('SDZ', self.subdivision[2]),
                Constant('IL_NO', self.il_array.shape[1]),
                Constant('IL_STRIDE_OUTER', self.il_array.shape[1] * self.il_array.shape[2]),
                Constant('NCOMP', self.ncomp),
                Constant('IM_OFFSET', self.L**2)
            )
        )

        dat_dict = {
            'P': self.positions(access.READ),
            'Q': self.charges(access.READ),
            'MM_FINE_CELLS': g._mm_fine_cells(access.WRITE),
            'MM_CELLS': g._mm_cells(access.WRITE),
            'MM_CHILD_INDEX': g._mm_child_index(access.WRITE),
            'OCC_GA': self.cell_occupation_ga(access.INC_ZERO),
            'TREE': self.tree(access.INC_ZERO),
            'WIDTHS_X': self.widths_x(access.READ),
            'WIDTHS_Y': self.widths_y(access.READ),
            'WIDTHS_Z': self.widths_z(access.READ),
            'NCELLS_X': self.ncells_x(access.READ),
            'NCELLS_Y': self.ncells_y(access.READ),
            'NCELLS_Z': self.ncells_z(access.READ),
        }

        self._contrib_loop = loop.ParticleLoopOMP(kernel=k, dat_dict=dat_dict)
    
        

    def _init_extract_loop(self):

        g = self.group
        extent = self.domain.extent
        cell_widths = [1.0 / (ex / (sx**(self.R - 1))) for ex, sx in zip(extent, self.subdivision)]


        sph_gen = self.sph_gen

        def cube_ind(L, M):
            return ((L) * ( (L) + 1 ) + (M) )


        assign_gen = ''
        for lx in range(self.L):
            for mx in range(-lx, lx+1):
                reL = SphSymbol('TREE[OFFSET + {ind}]'.format(ind=cube_ind(lx, mx)))
                imL = SphSymbol('TREE[OFFSET + IM_OFFSET + {ind}]'.format(ind=cube_ind(lx, mx)))
                reY, imY = sph_gen.get_y_sym(lx, mx)
                phi_sym = cmplx_mul(reL, imL, reY, imY)[0]
                assign_gen += 'tmp_energy += rhol * ({phi_sym});\n'.format(phi_sym=str(phi_sym))

            assign_gen += 'rhol *= iradius;\n'




        k = kernel.Kernel(
            'mm_extract_loop',
            r'''
            
            const double rx = P.i[0];
            const double ry = P.i[1];
            const double rz = P.i[2];

            double particle_energy = 0.0;

            int64_t LEVEL_OFFSETS[R];
            LEVEL_OFFSETS[0] = 0;
            int64_t nx = 1;
            int64_t ny = 1;
            int64_t nz = 1;
            for(int level=1 ; level<R ; level++ ){{
                int64_t nprev = nx * ny * nz * NCOMP;
                LEVEL_OFFSETS[level] = LEVEL_OFFSETS[level - 1] + nprev;
                nx *= SDX;
                ny *= SDY;
                nz *= SDZ;
            }}


            for( int level=0 ; level<R ; level++ ){{

                // cell on this level
                const int64_t cfx = MM_CELLS.i[level*3 + 0];
                const int64_t cfy = MM_CELLS.i[level*3 + 1];
                const int64_t cfz = MM_CELLS.i[level*3 + 2];

                // number of cells on this level
                const int64_t ncx = NCELLS_X[level];
                const int64_t ncy = NCELLS_Y[level];
                const int64_t ncz = NCELLS_Z[level];


                // child on this level
                const int64_t cix = MM_CHILD_INDEX.i[level * 3 + 0];
                const int64_t ciy = MM_CHILD_INDEX.i[level * 3 + 1];
                const int64_t ciz = MM_CHILD_INDEX.i[level * 3 + 2];
                const int64_t ci = cix + SDX * (ciy + SDY * ciz);

                const double wx = WIDTHS_X[level];
                const double wy = WIDTHS_Y[level];
                const double wz = WIDTHS_Z[level];

                
                // loop over IL for this child cell
                for( int ox=0 ; ox<IL_NO ; ox++){{
                    
                    
                    const int64_t ocx = cfx + IL[ci * IL_STRIDE_OUTER + ox * 3 + 0];
                    const int64_t ocy = cfy + IL[ci * IL_STRIDE_OUTER + ox * 3 + 1];
                    const int64_t ocz = cfz + IL[ci * IL_STRIDE_OUTER + ox * 3 + 2];

                    // free space for now
                    if (ocx < 0) {{continue;}}
                    if (ocy < 0) {{continue;}}
                    if (ocz < 0) {{continue;}}
                    if (ocx >= ncx) {{continue;}}
                    if (ocy >= ncy) {{continue;}}
                    if (ocz >= ncz) {{continue;}}

                    const int64_t lin_ind = ocx + NCELLS_X[level] * (ocy + NCELLS_Y[level] * ocz);

                    const double dx = rx - ((-HEX) + (0.5 * wx) + (ocx * wx));
                    const double dy = ry - ((-HEY) + (0.5 * wy) + (ocy * wy));
                    const double dz = rz - ((-HEZ) + (0.5 * wz) + (ocz * wz));

                    const double xy2 = dx * dx + dy * dy;
                    const double radius = sqrt(xy2 + dz * dz);
                    const double theta = atan2(sqrt(xy2), dz);
                    const double phi = atan2(dy, dx);
                    
                    const int64_t OFFSET = LEVEL_OFFSETS[level] + NCOMP * lin_ind;

                    {SPH_GEN}
                    const double iradius = 1.0 / radius;
                    double rhol = iradius;
                    double tmp_energy = 0.0;
                    {ASSIGN_GEN}

                    particle_energy += tmp_energy;
                    
                    //printf("%d %d %d\n",IL[ci * IL_STRIDE_OUTER + ox * 3 + 0], IL[ci * IL_STRIDE_OUTER + ox * 3 + 1], IL[ci * IL_STRIDE_OUTER + ox * 3 + 2]);
                    //printf("> %d | %f | %f %f %f | %d %d %d\n", ox, tmp_energy, dx, dy, dz, ocx, ocy, ocz);
                }}

                //printf("---> %d | %f | ci = %d\n", level, particle_energy, ci);

            }}

            
            //for(int ix=0 ; ix<4536 ; ix++){{
            //    printf("%d,", IL[ix]);
            //}}
            //printf("\n");


            OUT_ENERGY[0] += particle_energy * 0.5 * Q.i[0];

            '''.format(
                SPH_GEN=str(sph_gen.module),
                ASSIGN_GEN=str(assign_gen)
            ),
            (
                Constant('R', self.R),
                Constant('EX', extent[0]),
                Constant('EY', extent[1]),
                Constant('EZ', extent[2]),
                Constant('HEX', 0.5 * extent[0]),
                Constant('HEY', 0.5 * extent[1]),
                Constant('HEZ', 0.5 * extent[2]),                
                Constant('CWX', cell_widths[0]),
                Constant('CWY', cell_widths[1]),
                Constant('CWZ', cell_widths[2]),
                Constant('LCX', self.subdivision[0] ** (self.R - 1)),
                Constant('LCY', self.subdivision[1] ** (self.R - 1)),
                Constant('LCZ', self.subdivision[2] ** (self.R - 1)),
                Constant('SDX', self.subdivision[0]),
                Constant('SDY', self.subdivision[1]),
                Constant('SDZ', self.subdivision[2]),
                Constant('IL_NO', self.il_array.shape[1]),
                Constant('IL_STRIDE_OUTER', self.il_array.shape[1] * self.il_array.shape[2]),
                Constant('NCOMP', self.ncomp),
                Constant('IM_OFFSET', self.L**2)               
            )
        )

        dat_dict = {
            'P': self.positions(access.READ),
            'Q': self.charges(access.READ),
            'IL': self.il_scalararray(access.READ),
            'MM_CELLS': g._mm_cells(access.READ),
            'MM_CHILD_INDEX': g._mm_child_index(access.READ),
            'TREE': self.tree(access.READ),
            'WIDTHS_X': self.widths_x(access.READ),
            'WIDTHS_Y': self.widths_y(access.READ),
            'WIDTHS_Z': self.widths_z(access.READ),                            
            'NCELLS_X': self.ncells_x(access.READ),
            'NCELLS_Y': self.ncells_y(access.READ),
            'NCELLS_Z': self.ncells_z(access.READ),                     
            'OUT_ENERGY': self._extract_energy(access.INC_ZERO),
        }

        self._extract_loop = loop.ParticleLoopOMP(kernel=k, dat_dict=dat_dict)


    def __call__(self, positions, charges, forces=None, potential=None):
        if potential is not None or forces is not None:
            raise RuntimeError('Potential and Forces not implemented')

        self._contrib_loop.execute(
             dat_dict = {
                'P': positions(access.READ),
                'Q': charges(access.READ),
                'MM_FINE_CELLS': self.group._mm_fine_cells(access.WRITE),
                'MM_CELLS': self.group._mm_cells(access.WRITE),
                'MM_CHILD_INDEX': self.group._mm_child_index(access.WRITE),
                'OCC_GA': self.cell_occupation_ga(access.INC_ZERO),
                'TREE': self.tree(access.INC_ZERO),
                'WIDTHS_X': self.widths_x(access.READ),
                'WIDTHS_Y': self.widths_y(access.READ),
                'WIDTHS_Z': self.widths_z(access.READ),
                'NCELLS_X': self.ncells_x(access.READ),
                'NCELLS_Y': self.ncells_y(access.READ),
                'NCELLS_Z': self.ncells_z(access.READ),
            }               
        )

        self._extract_loop.execute(
            dat_dict = {
                'P': positions(access.READ),
                'Q': charges(access.READ),
                'IL': self.il_scalararray(access.READ),
                'MM_CELLS': self.group._mm_cells(access.READ),
                'MM_CHILD_INDEX': self.group._mm_child_index(access.READ),
                'TREE': self.tree(access.READ),
                'WIDTHS_X': self.widths_x(access.READ),
                'WIDTHS_Y': self.widths_y(access.READ),
                'WIDTHS_Z': self.widths_z(access.READ),                            
                'NCELLS_X': self.ncells_x(access.READ),
                'NCELLS_Y': self.ncells_y(access.READ),
                'NCELLS_Z': self.ncells_z(access.READ),                     
                'OUT_ENERGY': self._extract_energy(access.INC_ZERO),
            }
        )

        direct_energy = self._execute_direct(positions, charges, forces, potential)

        
        return self._extract_energy[0] + direct_energy



    def _profile_inc(self, key, inc):
        key = self.__class__.__name__ + ':' + key
        if key not in PROFILE.keys():
            PROFILE[key] = inc
        else:
            PROFILE[key] += inc








