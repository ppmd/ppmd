#!/usr/bin/python2.7

"""
Naive implementation of a system involving point objects with a Lennard-Jones potential.
"""


import numpy
#import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def lattice_init(ix, N, box_size):
    """
    Function to create a simple 3d lattice in provided volume. Returns a point in R^3.
    
    keyword arguments:
    
    ix -- linear index for a particle
    N -- Total number of particles
    box_size -- numpy.array containing volume dimensions
    
    """
    
    #Map point into cube side length R^(1/3)
    z=math.floor(ix/(N**(2.0/3.0)))
    y=math.floor((ix - z*(N**(2.0/3.0)))/(N**(1.0/3.0)))
    x=math.fmod((ix - z*(N**(2.0/3.0))),(N**(1.0/3.0)))
    
    #Crudly stretch cube to fit volume dimensions
    z=box_size[2,0]+(z/(N**(1.0/3.0)))*(box_size[2,1]-box_size[2,0])
    y=box_size[1,0]+(y/(N**(1.0/3.0)))*(box_size[1,1]-box_size[1,0])
    x=box_size[1,0]+(x/(N**(1.0/3.0)))*(box_size[0,1]-box_size[0,0])
    
    return [x,y,z]


class particle():
    """Base class containing infomation for one particle"""
    def __init__(self, loc=[0.0,0.0,0.0], m=1.0, vel=[0.0, 0.0, 0.0] ):
        """
        Initialise a particle'
        
        keyword arguments:
        loc -- list representing point in R^3, default [0.0, 0.0, 0.0].
        m -- mass of particle, default 1.0.
        vel -- list for velocity in R^3, default [0.0, 0.0, 0.0].
        """
        self.X=numpy.array([(loc[0]),(loc[1]),(loc[2])])
        self.V=numpy.array([(vel[0]),(vel[1]),(vel[2])])
        self.A=numpy.array([(0),(0),(0)])
        self.mass=m

class sim():
    """Object containing a simulation."""
    p_store=[]

    def __init__(self,N_in=0 , box_dims):
    """
    Initialise system,
    
    keyword arguments:
    N_in -- Number of particles, default 0.
    box_dims -- Dimensions of container, no default, must be specified.
    
    """
    
        self.N=N_in
        self.box_size=box_dims
        
        
        for i in range(N): #Create N new particles in a uniform grid
            loc=lattice_init(i,self.N, self.box_size)
            self.p_store.append(particle(loc))



    def verlet_vel(self, dt):
    """
    Perform velocity update in velocity verlet using specified timestep.
    """
        for i in self.p_store:
            i.V = i.V + dt*i.A

    def verlet_pos(self, dt):
    """
    Perform position update in velocity verlet using specified timestep. Applies peroidic boundary conditions
    """
        for i in self.p_store:
            i.X = i.X + dt*i.V #Position update from velocity
            
            for j in range(3): #Apply boundary conditions to each dimension
                if i.X[j] < self.box_size[j,0]:
                    i.X[j]= self.box_size[j,1] + i.X[j] - self.box_size[j,0]
                elif i.X[j] > self.box_size[j,1]:
                    i.X[j]= self.box_size[j,0] + i.X[j] - self.box_size[j,1]
            

    def accel_update(self):
        """
        Evaluate accelerations for all particles using Lennard-Jones potential. Assumes unit mass.
        """
        
        #Reset all accelerations to 0.
        for i in self.p_store:
            i.A=numpy.array([(0.0),(0.0),(0.0)])

        for i in range(self.N):
            for j in range(i+1,self.N):

                #Calculate direction
                a_tmp=numpy.subtract(self.p_store[i].X, self.p_store[j].X)
                
                r=numpy.linalg.norm(a_tmp)
                

                #Evaluate acceleration from LJ potential, assumes unit mass.
                if r>0:
                    a_tmp=a_tmp*(48*((1.0/r)**(14)) - 24*((1/r)**(8)) )
                elif r <= 0:
                    print 'r = 0 error'
                
                #Add acceleration to particles.
                self.p_store[i].A=numpy.add(self.p_store[i].A, a_tmp)
                self.p_store[j].A=numpy.subtract(self.p_store[j].A, a_tmp)
    
    def verlet_step(self, dt, it_max):
        """
        Perform a set number of velocity verlet integration steps.
        
        keyword arguments:
        dt -- Timestep size, no default.
        it_max -- number of timestep iterations to perform.
        """
        for i in range(it_max):
            self.verlet_vel(dt*0.5)
            self.verlet_pos(dt)
            self.accel_update()
            self.verlet_vel(dt*0.5)
            
            
    
    def frame_plot(self):
    """
    Function to plot all particles in 3D scatter plot.
    """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in self.p_store:
            #plt.scatter(i.X[0], i.X[1], alpha=0.5)
            ax.scatter(i.X[0], i.X[1], i.X[2])
        plt.show()




if __name__ == '__main__':
    """
    Program entry point
    """


    it_max=1   #max iterations
    N=27     #number of paricles
    box_size=numpy.array([(-5,5),(-5,5),(-5,5)])    #size of box containing particles
    dt=0.005

    #create simulation
    s1=sim(N,box_size)
    
    #integrate forwards in time
    s1.verlet_step(dt, it_max)
    s1.frame_plot()  

    

    























