"""
Multipole expansion based long range electrostatics O(NlogN).
"""
from __future__ import division, print_function, absolute_import

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


import numpy as np
from ppmd import data, loop, kernel, access, lib, opt, pairloop
from ppmd.coulomb import fmm_interaction_lists, mm_lm_common

from ppmd.coulomb.sph_harm import *

import ctypes
import math
import time

Constant = kernel.Constant
REAL = ctypes.c_double
INT64 = ctypes.c_int64

PROFILE = opt.PROFILE


BCType = mm_lm_common.BCType

class PyMM(mm_lm_common.MM_LM_Common):
    
    def _init_dats(self):
        g = self.group
        pd = type(self.charges)
        # xyz fmm cells
        g._mm_fine_cells = pd(ncomp=3, dtype=INT64)
        g._mm_cells = pd(ncomp=3*self.R, dtype=INT64)
        g._mm_child_index = pd(ncomp=3*self.R, dtype=INT64)

        self._dat_fine_cells = g._mm_fine_cells
        self._dat_cells = g._mm_cells
        self._dat_child_index = g._mm_child_index


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
            ),
            headers=(
                lib.build.write_header(
                    """
                    #define R {R}
                    const double WIDTHS_X[R] = {{ {WIDTHS_X} }};
                    const double WIDTHS_Y[R] = {{ {WIDTHS_Y} }};
                    const double WIDTHS_Z[R] = {{ {WIDTHS_Z} }};

                    const int64_t NCELLS_X[R] = {{ {NCELLS_X} }};
                    const int64_t NCELLS_Y[R] = {{ {NCELLS_Y} }};
                    const int64_t NCELLS_Z[R] = {{ {NCELLS_Z} }};

                    const int64_t LEVEL_OFFSETS[R] = {{ {LEVEL_OFFSETS} }};

                    """.format(
                        R=self.R,
                        WIDTHS_X=self.widths_x_str,
                        WIDTHS_Y=self.widths_y_str,
                        WIDTHS_Z=self.widths_z_str,
                        NCELLS_X=self.ncells_x_str,
                        NCELLS_Y=self.ncells_y_str,
                        NCELLS_Z=self.ncells_z_str,
                        LEVEL_OFFSETS=self.level_offsets_str
                    )
                ),
            )
        )

        dat_dict = {
            'P': self.positions(access.READ),
            'Q': self.charges(access.READ),
            'MM_FINE_CELLS': self._dat_fine_cells(access.WRITE),
            'MM_CELLS': self._dat_cells(access.WRITE),
            'MM_CHILD_INDEX': self._dat_child_index(access.WRITE),
            'OCC_GA': self.cell_occupation_ga(access.INC_ZERO),
            'TREE': self.tree(access.INC_ZERO),
        }

        self._contrib_loop = loop.ParticleLoopOMP(kernel=k, dat_dict=dat_dict)
        #self._contrib_loop = loop.ParticleLoop(kernel=k, dat_dict=dat_dict)
    
        

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


        bc = self.boundary_condition
        if bc == BCType.FREE_SPACE:
            bc_block = r'''
                if (ocx < 0) {{continue;}}
                if (ocy < 0) {{continue;}}
                if (ocz < 0) {{continue;}}
                if (ocx >= ncx) {{continue;}}
                if (ocy >= ncy) {{continue;}}
                if (ocz >= ncz) {{continue;}}
            '''
        elif bc in (BCType.NEAREST, BCType.PBC):
            bc_block = r'''
                ocx = (ocx + ({O})*ncx) % ncx;
                ocy = (ocy + ({O})*ncy) % ncy;
                ocz = (ocz + ({O})*ncz) % ncz;
            '''.format(O=self.max_il_offset*2)
        else:
            raise RuntimeError('Unkown boundary condition.')



        k = kernel.Kernel(
            'mm_extract_loop',
            r'''
            
            const double rx = P.i[0];
            const double ry = P.i[1];
            const double rz = P.i[2];

            double particle_energy = 0.0;


            for( int level=1 ; level<R ; level++ ){{

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
                    
                    
                    int64_t ocx = cfx + IL[ci * IL_STRIDE_OUTER + ox * 3 + 0];
                    int64_t ocy = cfy + IL[ci * IL_STRIDE_OUTER + ox * 3 + 1];
                    int64_t ocz = cfz + IL[ci * IL_STRIDE_OUTER + ox * 3 + 2];

                    const double dx = rx - ((-HEX) + (0.5 * wx) + (ocx * wx));
                    const double dy = ry - ((-HEY) + (0.5 * wy) + (ocy * wy));
                    const double dz = rz - ((-HEZ) + (0.5 * wz) + (ocz * wz));

                    {BC_BLOCK}

                    const int64_t lin_ind = ocx + NCELLS_X[level] * (ocy + NCELLS_Y[level] * ocz);

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
                    
                    //if (isnan(tmp_energy)){{
                    //    printf(
                    //        "radius %f theta %f phi %f\n",
                    //        radius, theta, phi
                    //    );
                    //    std::raise(SIGINT);
                    //}}
                    particle_energy += tmp_energy;
                    

                }}


            }}


            OUT_ENERGY[0] += particle_energy * 0.5 * Q.i[0];

            '''.format(
                SPH_GEN=str(sph_gen.module),
                ASSIGN_GEN=str(assign_gen),
                BC_BLOCK=bc_block
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
            ),
            headers=(
                lib.build.write_header(
                    """
                    #include <csignal>
                    #define R {R}
                    const double WIDTHS_X[R] = {{ {WIDTHS_X} }};
                    const double WIDTHS_Y[R] = {{ {WIDTHS_Y} }};
                    const double WIDTHS_Z[R] = {{ {WIDTHS_Z} }};

                    const int64_t NCELLS_X[R] = {{ {NCELLS_X} }};
                    const int64_t NCELLS_Y[R] = {{ {NCELLS_Y} }};
                    const int64_t NCELLS_Z[R] = {{ {NCELLS_Z} }};

                    const int64_t LEVEL_OFFSETS[R] = {{ {LEVEL_OFFSETS} }};

                    """.format(
                        R=self.R,
                        WIDTHS_X=self.widths_x_str,
                        WIDTHS_Y=self.widths_y_str,
                        WIDTHS_Z=self.widths_z_str,
                        NCELLS_X=self.ncells_x_str,
                        NCELLS_Y=self.ncells_y_str,
                        NCELLS_Z=self.ncells_z_str,
                        LEVEL_OFFSETS=self.level_offsets_str
                    )
                ),
            )
        )

        dat_dict = {
            'P': self.positions(access.READ),
            'Q': self.charges(access.READ),
            'IL': self.il_scalararray(access.READ),
            'MM_CELLS': self._dat_cells(access.READ),
            'MM_CHILD_INDEX': self._dat_child_index(access.READ),
            'TREE': self.tree(access.READ),
            'OUT_ENERGY': self._extract_energy(access.INC_ZERO),
        }

        self._extract_loop = loop.ParticleLoopOMP(kernel=k, dat_dict=dat_dict)
        #self._extract_loop = loop.ParticleLoop(kernel=k, dat_dict=dat_dict)








    def __call__(self, positions, charges, forces=None, potential=None):
        if potential is not None or forces is not None:
            raise RuntimeError('Potential and Forces not implemented')

        self._contrib_loop.execute(
             dat_dict = {
                'P': positions(access.READ),
                'Q': charges(access.READ),
                'MM_FINE_CELLS': self._dat_fine_cells(access.WRITE),
                'MM_CELLS': self._dat_cells(access.WRITE),
                'MM_CHILD_INDEX': self._dat_child_index(access.WRITE),
                'OCC_GA': self.cell_occupation_ga(access.INC_ZERO),
                'TREE': self.tree(access.INC_ZERO),
            }               
        )

        self._extract_loop.execute(
            dat_dict = {
                'P': positions(access.READ),
                'Q': charges(access.READ),
                'IL': self.il_scalararray(access.READ),
                'MM_CELLS': self._dat_cells(access.READ),
                'MM_CHILD_INDEX': self._dat_child_index(access.READ),
                'TREE': self.tree(access.READ),
                'OUT_ENERGY': self._extract_energy(access.INC_ZERO),
            }
        )

        direct_energy = self._execute_direct(positions, charges, forces, potential)

        pbc_energy = self._compute_pbc_contrib()

        # print('self._extract_energy[0]', 'direct_energy', 'pbc_energy',
        # self._extract_energy[0], direct_energy, pbc_energy)


        return self._extract_energy[0] + direct_energy + pbc_energy



