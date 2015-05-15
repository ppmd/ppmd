import numpy as np
import particle
import math
import ctypes
import time
import random
import pairloop
import data
import kernel
import loop
from mpi4py import MPI

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold='nan')

class BaseMDState(object):
    '''
    Base molecular dynamics class.
    
    :arg domain domain: Container within which the simulation takes place.
    :arg potential potential: Potential to use between particles.
    :arg PosInit* particle_pos_init: Class to initialise particles with.
    :arg VelInit* particle_vel_init: Class to initialise particles velocities with.
    :arg int N: Number of particles, default 1.
    :arg double mass: Mass of particles, default 1.0.
    :arg bool DEBUG: Flag to enable debug flags.
    '''
    def __init__(self, domain, potential, particle_pos_init = None, particle_vel_init = None, particle_mass_init = None, N = 0, mass = 1., DEBUG = False):
        '''
        Intialise class to hold the state of a simulation.
        :arg domain domain: Container within which the simulation takes place.
        :arg potential potential: Potential to use between particles.
        :arg int N: Number of particles, default 1.
        :arg double mass: Mass of particles, default 1.0        
        
        '''
              
        
        self._potential = potential
        self._N = N
        self._pos = particle.Dat(N, 3, name='positions')
        self._vel = particle.Dat(N, 3, name='velocities')
        self._accel = particle.Dat(N, 3, name='accelerations')
        
        self._mass = particle.Dat(N, 1, 1.0)
        if (particle_mass_init != None):
            particle_mass_init.reset(self._mass)
            
        
        
        self._domain = domain
        
        
        
        #potential energy, kenetic energy, total energy.
        self._U = data.ScalarArray();
        self._K = data.ScalarArray();
        self._Q = data.ScalarArray();

        
        
        ''' Initialise particle positions'''
        particle_pos_init.reset(self)
        self._domain.BCSetup(self._pos)
        
        '''Initialise velocities'''
        if (particle_vel_init != None):
            particle_vel_init.reset(self)
        
        
        '''Attempt to initialise cell array'''
        self._cell_setup_attempt = self._domain.set_cell_array_radius(self._potential._rn)
        
        
        
        print "Cell array = ", self._domain._cell_array
        print "Domain extents = ",self._domain._extent
        print "cell count:", self._domain.cell_count
        
        
        
        #Setup acceleration updating from given potential
        self._DEBUG = DEBUG
        _potential_dat_dict = self._potential.datdict(self)
        
        
        if (self._cell_setup_attempt==True):
            self._cell_sort_setup()
            
            
            self._cell_sort_all()
            #self._domain.halos.exchange(self._cell_contents_count, self._q_list, self._pos)            
            #self._cell_sort_all()
            
            
            
            
            self._looping_method_accel = pairloop.PairLoopRapaportHalo(N=self._N,
                                                                    domain = self._domain, 
                                                                    positions = self._pos, 
                                                                    potential = self._potential, 
                                                                    dat_dict = _potential_dat_dict,
                                                                    cell_list = self._q_list,
                                                                    DEBUG = self._DEBUG)
        
        else:
            self._looping_method_accel = pairloop.DoubleAllParticleLoopPBC(N=self._N,
                                                                        domain = self._domain, 
                                                                        kernel = self._potential.kernel,
                                                                        particle_dat_dict = _potential_dat_dict,
                                                                        DEBUG = self._DEBUG)
        
        self._time = 0
        
        
                                                                       
    def _cell_sort_setup(self):
        """
        Creates looping for cell list creation
        """
        
        '''Construct initial cell list'''
        self._q_list = data.ScalarArray(dtype=ctypes.c_int, max_size = self._N * (self._domain.cell_count) + self._domain.cell_count + 1)
        
        
        '''Keep track of number of particles per cell'''
        self._cell_contents_count = data.ScalarArray(np.zeros([self._domain.cell_count], dtype=ctypes.c_int, order='C'), dtype=ctypes.c_int)
        
        
        #temporary method for index awareness inside kernel.
        self._internal_index = data.ScalarArray(dtype=ctypes.c_int)
        self._internal_N = data.ScalarArray(dtype=ctypes.c_int)
        
        
        
        self._cell_sort_code = '''
        
        printf("cell start");
        
        const double R0 = P[0]+0.5*E[0];
        const double R1 = P[1]+0.5*E[1];
        const double R2 = P[2]+0.5*E[2];
        
        const int C0 = (int)(R0/CEL[0]);
        const int C1 = (int)(R1/CEL[1]);
        const int C2 = (int)(R2/CEL[2]);
        
        const int val = (C2*CA[1] + C1)*CA[0] + C0;
        
        printf("val=%d",val);
        
        CCC[val]++;
        
        Q[I[0]] = Q[N[0] + val];
        Q[N[0] + val] = I[0];
        I[0]++;
        '''
        self._cell_sort_dict = {'E':self._domain.extent,
                                'P':self._pos,
                                'CEL':self._domain.cell_edge_lengths,
                                'CA':self._domain.cell_array,
                                'Q':self._q_list,
                                'CCC':self._cell_contents_count,
                                'I':self._internal_index,
                                'N':self._internal_N}
                
        
        
        self._cell_sort_kernel = kernel.Kernel('cell_list_method', self._cell_sort_code, headers = ['stdio.h'])
        self._cell_sort_loop = loop.SingleParticleLoop(None, self._cell_sort_kernel, self._cell_sort_dict, DEBUG = self._DEBUG)
        
        
    def _cell_sort_all(self):
        """
        Construct neighbour list, assigning *all* atoms to cells. Using Rapaport algorithm.
        """
        self._q_list.resize(self._pos.npart + self._pos.npart_halo + self._domain.cell_count + 1)
        self._q_list[self._q_list.end] = self._q_list.end - self._domain.cell_count
        self._internal_N[0] = self._q_list[self._q_list.end]
        
        self._q_list.Dat[self._q_list[self._q_list.end]:self._q_list.end:] = ctypes.c_int(-1)
        
        
        self._internal_index[0]=0
        self._cell_contents_count.zero()
        self._internal_N[0] = self._q_list[self._q_list.end]
        self._cell_sort_loop.execute(start = 0, end=self._pos.npart + self._pos.npart_halo, dat_dict = {'E':self._domain.extent,
                                'P':self._pos,
                                'CEL':self._domain.cell_edge_lengths,
                                'CA':self._domain.cell_array,
                                'Q':self._q_list,
                                'CCC':self._cell_contents_count,
                                'I':self._internal_index,
                                'N':self._internal_N})
        
        
         
    def _cell_sort_local(self):
        """
        Construct neighbour list, assigning *local* atoms to cells. Using Rapaport algorithm.
        """
        
        self._q_list.resize(self._pos.npart + self._pos.npart_halo + self._domain.cell_count + 1)
        self._q_list[self._q_list.end] = self._q_list.end - self._domain.cell_count
        
        self._internal_N[0] = self._q_list[self._q_list.end]
        self._q_list.Dat[self._q_list[self._q_list.end]:self._q_list.end:] = ctypes.c_int(-1)
        self._internal_index[0]=0
        
        self._cell_contents_count.zero()
        self._internal_N[0] = self._q_list[self._q_list.end]
        self._cell_sort_loop.execute(   start = 0, 
                                        end=self._pos.npart,
                                        dat_dict = {'E':self._domain.extent,
                                                    'P':self._pos,
                                                    'CEL':self._domain.cell_edge_lengths,
                                                    'CA':self._domain.cell_array,
                                                    'Q':self._q_list,
                                                    'CCC':self._cell_contents_count,
                                                    'I':self._internal_index,
                                                    'N':self._internal_N}
                                     )    
        
          
        
        
    def _cell_sort_halo(self):
        """
        Construct neighbour list, assigning *halo* atoms to cells. Using Rapaport algorithm. Depreciated, use halo sorting methods within the halo class.
        """
        
        self._q_list.resize(self._pos.npart + self._pos.npart_halo + self._domain.cell_count + 1)
        self._q_list[self._q_list.end] = self._q_list.end - self._domain.cell_count
        self._internal_N[0] = self._q_list[self._q_list.end]
        
        
        self._internal_index[0]=self._pos.npart
        self._internal_N[0] = self._q_list[self._q_list.end]
        self._cell_sort_loop.execute(   start = self._pos.npart, 
                                        end=self._pos.npart + self._pos.npart_halo,
                                        dat_dict = {'E':self._domain.extent,
                                                    'P':self._pos,
                                                    'CEL':self._domain.cell_edge_lengths,
                                                    'CA':self._domain.cell_array,
                                                    'Q':self._q_list,
                                                    'CCC':self._cell_contents_count,
                                                    'I':self._internal_index,
                                                    'N':self._internal_N}
                                     )          
        
        
    @property    
    def N(self):
        """
        Returns number of particles.
        """
        return self._N
            
        
        
    
    @property  
    def domain(self):
        """
        Return the domain used by the state.
        """
        return self._domain
        
    @property    
    def positions(self):
        """
        Return all particle positions.
        """
        return self._pos

    @property    
    def velocities(self):
        """
        Return all particle velocities.
        """
        return self._vel
    
    @property     
    def forces(self):
        """
        Return all particle forces.
        """
        return self._accel

    @property      
    def masses(self):
        """
        Return all particle masses.
        """
        return self._mass
        
    def set_forces(self,val):
        """
        Set all forces to given value.
        
        :arg double val: value to set to.
        """
        
        #self._accel.set_val(val)
        self._accel.Dat[0:self._accel.npart:,::] = val
        
        
        
    def forces_update(self):
        """
        Updates forces dats using given looping method.
        """
        timer = True         
        if (timer==True):
            start = time.time() 
        
        print "start local cell sort"
        print self._q_list.ctypes_data
        self._cell_sort_local()               
        print "end local cell sort"
        
        
        if (self._cell_setup_attempt==True):
            self._domain.halos.exchange(self._cell_contents_count, self._q_list, self._pos)
        
        
        self.set_forces(ctypes.c_double(0.0))
        self.reset_U()
        #const int n, int* cell_array, int* q_list, double* d_extent,%(ARGUMENTS)s
        
        
        
        self._looping_method_accel.execute()   
        
        if (timer==True):
            end = time.time()
            self._time+=end - start       
        
    @property
    def potential(self):
        return self._potential
             
        
    @property    
    def U(self):
        """
        Return potential energy
        """
        return self._U
    
    @property    
    def K(self):
        """
        Return Kenetic energy
        """
        return self._K
    
    @property    
    def Q(self):
        """
        Return Total energy
        """
        return self._Q        
                
        
    def reset_U(self):
        """
        Reset potential energy to 0.0
        """
        self._U._Dat = np.zeros([1], dtype=ctypes.c_double, order='C')
        
        
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
        

        
        
        
class PosInitLatticeNRho(object):
    """
    Arrange N particles into a 3D lattice of density :math:`/rho`. Redfines container volume as a cube with deduced volume, assumes unit mass.
    
    :arg int N: N, number of particles.
    :arg double rho: :math:`rho`, required density.
    
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
        
        :arg state state_input: object of state class. Inheritered from BaseMDState.
        """
        
        #Evaluate cube side length.
        Lx = (float(self._N) / float(self._rho))**(1./3.)
        
        #Cube dimensions of data
        np1_3 = self._N**(1./3.)
        np2_3 = np1_3**2.
        
        #starting point for each dimension. 
        mLx_2 = (-0.5 * Lx) + (0.5*Lx)/math.floor(np1_3)
        
        #set new domain extents
        state_input.domain.set_extent(np.array([Lx, Lx, Lx]))
        
        #get pointer for positions
        pos = state_input.positions
        
        #Loop over all particles
        for ix in range(self._N):
            
            #Map point into cube side of calculated side length Lx.
            z=math.floor(ix/np2_3)

            pos[ix,0]=mLx_2+(math.fmod((ix - z*np2_3),np1_3)/np1_3)*Lx #x
            pos[ix,1]=mLx_2+(math.floor((ix - z*np2_3)/np1_3)/np1_3)*Lx #y
            pos[ix,2]=mLx_2+(z/np1_3)*Lx
            
        
        
class PosInitLatticeNRhoRand(object):
    """
    Arrange N particles into a 3D lattice of density :math:`/rho`. Redfines container volume as a cube with deduced volume, assumes unit mass adds uniform deviantion based on given maximum.
    
        :arg int N: number of particles.
        :arg double rho: :math:`/rho`, required density.
        :arg double dev: maximum possible random deviation from lattice.
    
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
        
        :arg state state_input: object of state class. Inheritered from BaseMDState.
        """
        
        #Evaluate cube side length.
        Lx = (float(self._N) / float(self._rho))**(1./3.)
        
        #Cube dimensions of data
        np1_3 = self._N**(1./3.)
        np2_3 = np1_3**2.
        
        #starting point for each dimension. 
        mLx_2 = (-0.5 * Lx) + (0.5*Lx)/math.floor(np1_3)
        
        #set new domain extents
        state_input.domain.set_extent(np.array([Lx, Lx, Lx]))
        
        #get pointer for positions
        pos = state_input.positions
        
        #Loop over all particles
        for ix in range(self._N):
            
            #Map point into cube side of calculated side length Lx.
            z=math.floor(ix/np2_3)

            pos[ix,0]=random.uniform(0,self._dev) + mLx_2+(math.fmod((ix - z*np2_3),np1_3)/np1_3)*Lx #x
            pos[ix,1]=random.uniform(0,self._dev) + mLx_2+(math.floor((ix - z*np2_3)/np1_3)/np1_3)*Lx #y
            pos[ix,2]=random.uniform(0,self._dev) + mLx_2+(z/np1_3)*Lx


class PosInitTwoParticlesInABox(object):
    """
    Creates two particles a set distance apart on the  given axis, centred on the origin. Places these within a containing volume of given extents.
    
    :arg double rx: Distance between particles.
    :arg np.array(3,1) extents: Extent for containing volume.
    :arg np.array(3,1) axis: axis to centre on.
    """
    
    def __init__(self,rx,extent = np.array([1.0,1.0,1.0]), axis = np.array([1.0,0.0,0.0])):
        self._extent = extent
        self._axis = axis
        self._rx = (0.5/np.linalg.norm(self._axis))*rx
        
    def reset(self, state_input):
        """
        Resets the first two particles in the input state domain to sit on the x-axis the set distance apart.
        
        
        :arg state state_input: State object containing at least two particles.
        """
        
        if (state_input.N >= 2):
            state_input.positions[0,] = -1.*self._rx*self._axis
            state_input.positions[1,] = self._rx*self._axis
        else:
            print "ERROR: PosInitTwoParticlesInABox, not enough particles!"
            
        state_input.domain.set_extent(self._extent)



class PosInitDLPOLYConfig(object):
    """
    Read positions from DLPLOY config file.
    
    :arg str filename: Config filename.
    """
    
    def __init__(self,filename = None):
        self._f = filename
        assert self._f!=None, "No position config file specified"
        
        
    def reset(self, state_input):
        """
        Resets particle positions to those in file.
        
        :arg state state_input: State object containing required number of particles.
        """
        
        fh=open(self._f)
        shift = 7
        offset= 4
        count = 0
        
        extent=np.array([0.,0.,0.])
        
        for i, line in enumerate(fh):
            if (i==2):
                extent[0]=line.strip().split()[0]
            if (i==3):
                extent[1]=line.strip().split()[1]                
            if (i==4):
                extent[2]=line.strip().split()[2]                
                
        
            if ((i>(shift-2)) and ((i-shift+1)%offset == 0) and count < state_input.N ):
                state_input.positions[count,0]=line.strip().split()[0]
                state_input.positions[count,1]=line.strip().split()[1]
                state_input.positions[count,2]=line.strip().split()[2]
                count+=1
        
        
        state_input.domain.set_extent(extent)







        
class VelInitNormDist(object):
    """
    Initialise velocities by sampling from a gaussian distribution.
    
    :arg double mu: Mean for gaussian distribution.
    :arg double sig: Standard deviation for gaussian distribution.
    
    """

    def __init__(self,mu = 0.0,sig = 1.0):
        self._mu = mu
        self._sig = sig        
    
    
    def reset(self,state_input):
        """
        Resets particle velocities to Gaussian distribution.
        
        :arg state state_input: Input state class oject containing velocities.
        """
        
        #Get velocities.
        vel_in = state_input.velocities
        
        #Apply normal distro to velocities.
        for ix in range(state_input.N):
            vel_in[ix,]=[random.gauss(self._mu, self._sig),random.gauss(self._mu, self._sig),random.gauss(self._mu, self._sig)]
        
        
class VelInitTwoParticlesInABox(object):
    """
    Sets velocities for two particles.
    
    :arg np.array(3,1) vx: Velocity vector for particle 1.
    :arg np.array(3,1) vy: Velocity vector for particle 2.
    
    """

    def __init__(self, vx = np.array([0., 0., 0.]), vy = np.array([0., 0., 0.])):
        self._vx = vx
        self._vy = vy
        
    def reset(self, state_input):
        """
        Resets the particles in the input state to the required velocities.
        
        :arg state state_input: input state.
        """

        if (state_input.N >= 2):
            state_input.velocities[0,] = self._vx
            state_input.velocities[1,] = self._vy        
        else:
            print "ERROR: PosInitTwoParticlesInABox, not enough particles!"


class VelInitMaxwellBoltzmannDist(object):
    """
    Initialise velocities by sampling from a gaussian distribution.
    
    :arg double mu: Mean for gaussian distribution.
    :arg double sig: Standard deviation for gaussian distribution.
    
    """

    def __init__(self,temperature=293.15):
        self._t = (float)(temperature)
        print "Warning not yet functional"
    
    
    def reset(self,state_input):
        """
        Resets particle velocities to Maxwell-Boltzmann distribution.
        
        :arg state state_input: Input state class oject containing velocities and masses.
        """
        
        #Apply MB distro to velocities.
        for ix in range(state_input.N):
            scale = math.sqrt(self._t/state_input.masses[ix])
            stmp = scale*math.sqrt(-2.0*math.log(random.uniform(0,1)))
            V0 = 2.*math.pi*random.uniform(0,1);
            state_input.velocities[ix,0]=stmp*math.cos(V0)
            state_input.velocities[ix,1]=stmp*math.sin(V0)
            state_input.velocities[ix,1]=scale*math.sqrt(-2.0*math.log(random.uniform(0,1)))*math.cos(2.*math.pi*random.uniform(0,1));
            

class VelInitDLPOLYConfig(object):
    """
    Read velocities from DLPLOY config file.
    
    :arg str filename: Config filename.
    """
    
    def __init__(self,filename = None):
        self._f = filename
        assert self._f!=None, "No position config file specified"
        
        
    def reset(self, state_input):
        """
        Resets particle velocities to those in file.
        
        :arg state state_input: State object containing required number of particles.
        """
        
        fh=open(self._f)
        shift = 8
        offset= 4
        count = 0
        
        
        for i, line in enumerate(fh):
            if ((i>(shift-2)) and ((i-shift+1)%offset == 0) and count < state_input.N ):
                state_input.velocities[count,0]=line.strip().split()[0]
                state_input.velocities[count,1]=line.strip().split()[1]
                state_input.velocities[count,2]=line.strip().split()[2]
                count+=1
        
        

class MassInitTwoAlternating(object):
    '''
    Class to initialise masses, alternates between two masses.
    
    :arg double m1:  First mass
    :arg double m2:  Second mass
    '''
    
    def __init__(self, m1 = 1.0, m2 = 1.0):
        self._m = [m1, m2]

        
    def reset(self, mass_input):
        '''
        Apply to input mass dat class.
        
        :arg Dat mass_input: Dat container with masses.
        '''
        for ix in range(np.shape(mass_input.Dat)[0]):
            mass_input[ix] = self._m[(ix % 2)]

class MassInitIdentical(object):
    '''
    Class to initialise all masses to one value.
    
    :arg double m: Mass default 1.0
    '''
    
    def __init__(self, m = 1.0):
        self._m = (float)(m)

        
    def reset(self, mass_input):
        '''
        Apply to input mass dat class.
        
        :arg Dat mass_input: Dat container with masses.
        '''
        for ix in range(mass_input.npart):
            mass_input[ix] = self._m











    
    
        
        
        
        
        
        
        
