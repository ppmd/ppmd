import numpy as np
import particle
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class BaseMDState():
    """
    Base molecular dynamics class, stores:
    """
    def __init__(self, domain, potential, particle_init = None, N = 0, mass = 1.):
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
        
        particle_init.reset(self)
        
        
        
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
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for ix in range(self._N):
            ax.scatter(self._pos[ix,0], self._pos[ix,1], self._pos[ix,2])
        plt.show()
        
        
class LatticeInitNRho():
    """
    Arrange N particles into a 3D lattice of density :math:`/rho`. Redfines container volume as a cube with deduced volume, assumes unit mass.
    
    """
    
    def __init__(self, N, rho):
        """
        Initialise required lattice with the number of particles and required density.
        
        :arg: (int) input, N, number of particles.
        :arg: (float) input, :math:`\rho`, required density.
       
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
            
        
        
        
        
        
        
        
        
        
    
    
        
        
        
        
        
        
        
