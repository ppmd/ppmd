import numpy as np
import particle
import math
import ctypes
import time
import random
import pairloop
import data

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold='nan')

class BaseMDState():
    """
    Base molecular dynamics class.
    
        :arg domain: (Domain class) Container within which the simulation takes place.
        :arg potential: (Potential class) Potential to use between particles.
        :arg particle_pos_init: (class PosInit*) Class to initialise particles with.
        :arg particle_vel_init: (class VelInit*) Class to initialise particles velocities with.
        :arg N: (integer) Number of particles, default 1.
        :arg mass: (float) Mass of particles, default 1.0
        :arg dt: (float) Time-step size.
        :arg T: (float) End time.
    """
    def __init__(self, domain, potential, particle_pos_init = None, particle_vel_init = None, N = 0, mass = 1.):
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
        

        

        

        
        #potential energy, kenetic energy, total energy.
        self._U = np.zeros([1], dtype=ctypes.c_double, order='C')
        self._K = np.zeros([1], dtype=ctypes.c_double, order='C')
        self._Q = np.zeros([1], dtype=ctypes.c_double, order='C')

        
        
        ''' Initialise particle positions'''
        particle_pos_init.reset(self)
        
        
        '''Initialise velocities'''
        if (particle_vel_init != None):
            particle_vel_init.reset(self)
        
        
        '''Initialise cell array'''
        self._domain.set_cell_array_radius(self._potential._rn)
        
        print "Cell array = ", self._domain._cell_array
        print "Domain extents = ",self._domain._extent
        
        
    
        
    def N(self):
        """
        Returns number of particles.
        """
        return self._N
        
    def domain(self):
        """
        Return the domain used by the state.
        """
        return self._domain
        
        
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
        
    def set_accelerations(self,val):
        """
        Set all accelerations to given value.
        
        :arg val: (float) value to set to.
        """
        
        self._accel.set_val(val)
        
    def U(self):
        """
        Return potential energy
        """
        return self._U
        
    def K(self):
        """
        Return Kenetic energy
        """
        return self._K
        
    def Q(self):
        """
        Return Total energy
        """
        return self._Q        
                
        
    def reset_U(self):
        """
        Reset potential energy to 0.0
        """
        self._U = 0.0
        
    def K_set(self, K_in):
        """
        Set a kenetic energy value.
        """
        self._K = K_in
        
    def U_set(self, U_in):
        """
        Set a kenetic energy value.
        """
        self._U = U_in
        
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
        

        
        
    def frame_plot_pos(self):
        """
        Function to plot all particles in 3D scatter plot.
        """
        
        print "plotting....."
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for ix in range(self._N):
            ax.scatter(self._pos[ix,0], self._pos[ix,1], self._pos[ix,2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        plt.show()
        
        
class PosInitLatticeNRho():
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
        state_input.domain().set_extent(np.array([Lx, Lx, Lx]))
        
        #get pointer for positions
        pos = state_input.positions()
        
        #Loop over all particles
        for ix in range(self._N):
            
            #Map point into cube side of calculated side length Lx.
            z=math.floor(ix/np2_3)

            pos[ix,0]=mLx_2+(math.fmod((ix - z*np2_3),np1_3)/np1_3)*Lx #x
            pos[ix,1]=mLx_2+(math.floor((ix - z*np2_3)/np1_3)/np1_3)*Lx #y
            pos[ix,2]=mLx_2+(z/np1_3)*Lx
            
        
        
class PosInitLatticeNRhoRand():
    """
    Arrange N particles into a 3D lattice of density :math:`/rho`. Redfines container volume as a cube with deduced volume, assumes unit mass adds uniform deviantion based on given maximum.
    
        :arg N: (int) number of particles.
        :arg rho: (float) :math:`/rho`, required density.
        :arg dev: (float) maximum possible random deviation from lattice.
    
    """
    
    def __init__(self, N, rho, dev=0.0):
        """
        Initialise required lattice with the number of particles and required density.
        
        """
        
        self._N = N
        self._rho = rho
        self._dev = dev

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
        state_input.domain().set_extent(np.array([Lx, Lx, Lx]))
        
        #get pointer for positions
        pos = state_input.positions()
        
        #Loop over all particles
        for ix in range(self._N):
            
            #Map point into cube side of calculated side length Lx.
            z=math.floor(ix/np2_3)

            pos[ix,0]=random.uniform(0,self._dev) + mLx_2+(math.fmod((ix - z*np2_3),np1_3)/np1_3)*Lx #x
            pos[ix,1]=random.uniform(0,self._dev) + mLx_2+(math.floor((ix - z*np2_3)/np1_3)/np1_3)*Lx #y
            pos[ix,2]=random.uniform(0,self._dev) + mLx_2+(z/np1_3)*Lx


class PosInitTwoParticlesInABox():
    """
    Creates two particles a set distance apart on the  given axis, centred on the origin. Places these within a containing volume of given extents.
    
    :arg rx: (float) Distance between particles.
    :arg extents: (np.array(3)) Extent for containing volume.
    :arg axis: (np.array(3)) axis to centre on.
    """
    
    def __init__(self,rx,extent = np.array([1.0,1.0,1.0]), axis = np.array([1.0,0.0,0.0])):
        self._rx = rx
        self._extent = extent
        self._axis = axis
        
    def reset(self, state_input):
        """
        Resets the first two particles in the input state domain to sit on the x-axis the set distance apart.
        
        
        :arg state_input: (state class) State object containing at least two particles.
        """
        
        if (state_input.N() >= 2):
            state_input.positions()[0,] = -0.5*self._rx*self._axis
            state_input.positions()[1,] = 0.5*self._rx*self._axis
        else:
            print "ERROR: PosInitTwoParticlesInABox, not enough particles!"
            
        state_input.domain().set_extent(self._extent)




        
class VelInitNormDist():
    """
    Initialise velocities by sampling from a gaussian distribution.
    
    :arg mu: (float) Mean for gaussian distribution.
    :arg sig: (float) Standard deviation for gaussian distribution.
    
    """

    def __init__(self,mu = 0.0,sig = 1.0):
        self._mu = mu
        self._sig = sig        
    
    
    def reset(self,state_input):
        """
        Resets particle velocities to Gaussian distribution.
        
        :arg state_input: (state class) Input state class oject containing velocities.
        """
        
        #Get velocities.
        vel_in = state_input.velocities()
        
        #Apply normal distro to velocities.
        for ix in range(state_input.N()):
            vel_in[ix,]=[random.gauss(self._mu, self._sig),random.gauss(self._mu, self._sig),random.gauss(self._mu, self._sig)]
        
        
class VelInitTwoParticlesInABox():
    """
    Sets velocities for two particles.
    
    :arg vx: (np.array(3,1)) Velocity vector for particle 1.
    :arg vy: (np.array(3,1)) Velocity vector for particle 2.
    
    """

    def __init__(self, vx = np.array([0., 0., 0.]), vy = np.array([0., 0., 0.])):
        self._vx = vx
        self._vy = vy
        
    def reset(self, state_input):
        """
        Resets the particles in the input state to the required velocities.
        
        :arg state_input: (state class) input state.
        """

        if (state_input.N() >= 2):
            state_input.velocities()[0,] = self._vx
            state_input.velocities()[1,] = self._vy        
        else:
            print "ERROR: PosInitTwoParticlesInABox, not enough particles!"

















    
    
        
        
        
        
        
        
        
