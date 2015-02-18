import numpy as np
import particle
import math
import ctypes
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class BaseMDState():
    """
    Base molecular dynamics class.
    
        :arg domain: (Domain class) Container within which the simulation takes place.
        :arg potential: (Potential class) Potential to use between particles.
        :arg N: (integer) Number of particles, default 1.
        :arg mass: (float) Mass of particles, default 1.0
    """
    def __init__(self, domain, potential, particle_init = None, N = 0, mass = 1., dt = 0.001, T = 0.04):
        """
        Intialise class to hold the state of a simulation.
        :arg domain: (Domain class) Container within which the simulation takes place.
        :arg potential: (Potential class) Potential to use between particles.
        :arg N: (integer) Number of particles, default 1.
        :arg mass: (float) Mass of particles, default 1.0        
        
        """        
        self._N = N
        self._pos = particle.Dat(N, 3)
        self._vel = particle.Dat(N, 3)
        self._accel = particle.Dat(N, 3)
        
        self._mass = particle.Dat(N, 1, 1.0)
        
        self._domain = domain
        self._potential = potential
        
        #potential energy
        self._U = np.zeros([1], dtype=ctypes.c_double, order='C')
        
        
        
        self._rc = 2.**(1./6.)*self._potential._sigma
        self._rn = 2*self._rc
        
        print "r_n = ", self._rn
        
        self._rn2 = self._rn**2
        self._N_p = 0
        
        self._dt = dt
        self._T = T

        
        ''' Initialise particle positions'''
        particle_init.reset(self)
        
        '''Initialise cell array'''
        self._domain.set_cell_array_radius(self._rn)
        
        print "Cell array = ", self._domain._cell_array
        print "Domain extents = ",self._domain._extent
        
        
        
        
        '''Construct initial cell list'''
        self._q_list = np.zeros([1 + self._N + self._domain.cell_count()], dtype=ctypes.c_int, order='C')
        self.cell_sort_all()
        
        
        """
        #Create pair lists and evaluate forces
        start = time.time()
        self.pair_locate_c()
        print "Potential energy C:", self._U
        end = time.time()
        self.pair_locate()
        end2 = time.time()
        print "Potential energy py:", self._U
        
        print "Time taken C: ", end - start, "seconds."
        print "Time taken py: ", end2 - end, "seconds."
        """
        
        #Calculate initial accelerations.
        self.pair_locate_c()
        
        self.velocity_verlet_integration()
        
        
        
        
    def velocity_verlet_step(self):
        """
        Perform one step of Velocity Verlet.
        """
        
        self._vel._Dat+=0.5*self._dt*self._accel._Dat
        self._pos._Dat+=self._dt*self._vel._Dat
        
        #handle perodic bounadies
        self._domain.boundary_correct(self._pos,self._N)
        
        #update accelerations
        self.pair_locate_c()
        self._vel._Dat+= 0.5*self._dt*self._accel._Dat
        
        
    def velocity_verlet_integration(self):
        """
        Perform Velocity Verlet integration up to time T.
        """    
        
        for i in range(int(math.ceil(self._T/self._dt))):
            self.velocity_verlet_step()
            print i

        
        
        
        
        
        
        
        
    def cell_sort_all(self):
        """
        Construct neighbour list, assigning atoms to cells. Using Rapaport alg.
        """
        for cx in range(1,1+self._domain.cell_count()):
            self._q_list[self._N + cx] = 0
        for ix in range(1,1+self._N):
            c = self._domain.get_cell_lin_index(self._pos[ix-1,])
            self._q_list[ix] = self._q_list[self._N + c]
            self._q_list[self._N + c] = ix
        
        
        
    def pair_locate(self):
        """
        Loop over all cells update accelerations and potential engery.
        """
        
        self._accel._Dat*=0.0
        self._U[0] = 0.0
        
        
        for cp in range(1,1 + self._domain.cell_count()):
            
            cells = self._domain.get_adjacent_cells(cp)
            
            count=0
            
            """start c code here"""
            for cpp in range(1,15):
                
                ip = self._q_list[self._N+cp]
                
                while (ip > 0):
                    ipp = self._q_list[self._N+cpp]
                    while (ipp > 0):
                        if (cp != cpp or ip < ipp):
                            #distance
                            
                            rv = self._pos[ip-1] - self._pos[ipp-1] + cells[cpp-1,1:4:1]*self._domain._extent
                            r = np.linalg.norm(rv)
                            

                            if (r**2 < (self._rc**2)):    

                                count+=1
                                
                                force_eval = self._potential.evaluate_force(r)
                                
                                self._accel._Dat[ip-1]+=force_eval*rv
                                self._accel._Dat[ipp-1]-=force_eval*rv
                                
                                self._U[0] += self._potential.evaluate(r)
                                
                                
                                      
                        ipp = self._q_list[ipp]
                    ip = self._q_list[ip]
            #print count, cp
        
        
    def pair_locate_c(self):
        """
        C version of the pair_locate: Loop over all cells update accelerations and potential engery.
        """
        
        
        self._accel._Dat*=0.0
        self._U[0] = 0.0
        
        
        
        self._libpair_loop_LJ = np.ctypeslib.load_library('libpair_loop_LJ.so','.')
        self._libpair_loop_LJ.d_pair_loop_LJ.restype = ctypes.c_int
        
        #void d_pair_loop_LJ(int N, int cp, double rc, int* cells, int* q_list, double* pos, double* d_extent, double *accel);
        self._libpair_loop_LJ.d_pair_loop_LJ.argtypes = [ctypes.c_int,
                                                        ctypes.c_int,
                                                        ctypes.c_double,
                                                        ctypes.POINTER(ctypes.c_int),
                                                        ctypes.POINTER(ctypes.c_int),
                                                        ctypes.POINTER(ctypes.c_double),
                                                        ctypes.POINTER(ctypes.c_double),
                                                        ctypes.POINTER(ctypes.c_double),
                                                        ctypes.POINTER(ctypes.c_double)]
        
        
        for cp in range(1,1 + self._domain.cell_count()):
            
            
            cells = self._domain.get_adjacent_cells(cp)
            
            
            args = [cells.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    self._q_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    self._pos._Dat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    self._domain._extent.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    self._accel._Dat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    self._U.ctypes.data_as(ctypes.POINTER(ctypes.c_double))]
            
            self._libpair_loop_LJ.d_pair_loop_LJ(ctypes.c_int(self._N), ctypes.c_int(cp), ctypes.c_double(self._rc), *args)
        
        
        
        
    def positions(self):
        """
        Return all particle positions.
        """
        return self._pos

        
    def velocities(self):
        """
        Return all particle velocities.
        """
        return self._vel
        
    def accelerations(self):
        """
        Return all particle accelerations.
        """
        return self._accel
        
    def energy_kenetic(self):
        """
        Calcluate and return kenetic energy.
        """
        
        def squared_sum(x): return (x[0]**2) + (x[1]**2) + (x[2]**2)
        
        print self._N
        
        energy = 0.
        for i in range(self._N):
            
            energy += 0.5*self._mass[i]*squared_sum(self._vel[i,:])
            
        return float(energy)
        
    def frame_plot(self):
        """
        Function to plot all particles in 3D scatter plot.
        """
        
        print "plotting....."
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for ix in range(self._N):
            ax.scatter(self._pos[ix,0], self._pos[ix,1], self._pos[ix,2])
        plt.show()
        
        
class LatticeInitNRho():
    """
    Arrange N particles into a 3D lattice of density :math:`/rho`. Redfines container volume as a cube with deduced volume, assumes unit mass.
    
        :arg: (int) input, N, number of particles.
        :arg: (float) input, :math:`/rho`, required density.
    
    """
    
    def __init__(self, N, rho):
        """
        Initialise required lattice with the number of particles and required density.
        
       
        """
        
        self._N = N
        self._rho = rho

    def reset(self, state_input):
        """
        Applies initial lattice to particle positions.
        
        :arg: (state.*) object of state class. Inheritered from BaseMDState.
        """
        
        #Evaluate cube side length.
        Lx = (float(self._N) / float(self._rho))**(1./3.)
        
        #Cube dimensions of data
        np1_3 = self._N**(1./3.)
        np2_3 = np1_3**2.
        
        #starting point for each dimension. 
        mLx_2 = (-0.5 * Lx) + (0.5*Lx)/math.floor(np1_3)
        
        #set new domain extents
        state_input._domain.set_extent(np.array([Lx, Lx, Lx]))
        
        #get pointer for positions
        pos = state_input.positions()
        
        #Loop over all particles
        for ix in range(self._N):
            
            #Map point into cube side of calculated side length Lx.
            z=math.floor(ix/np2_3)

            pos[ix,0]=mLx_2+(math.fmod((ix - z*np2_3),np1_3)/np1_3)*Lx #x
            pos[ix,1]=mLx_2+(math.floor((ix - z*np2_3)/np1_3)/np1_3)*Lx #y
            pos[ix,2]=mLx_2+(z/np1_3)*Lx
            
        
        
        
        
        
        
        
        
        
    
    
        
        
        
        
        
        
        
