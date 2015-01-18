#!/usr/bin/python2.7
import numpy
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def lattice_init(ix, N, box_size):
    z=math.floor(ix/(N**(2.0/3.0)))
    y=math.floor((ix - z*(N**(2.0/3.0)))/(N**(1.0/3.0)))
    x=math.fmod((ix - z*(N**(2.0/3.0))),(N**(1.0/3.0)))
    
    z=box_size[2,0]+(z/(N**(1.0/3.0)))*(box_size[2,1]-box_size[2,0])
    y=box_size[1,0]+(y/(N**(1.0/3.0)))*(box_size[1,1]-box_size[1,0])
    x=box_size[1,0]+(x/(N**(1.0/3.0)))*(box_size[0,1]-box_size[0,0])
    
    return [x,y,z]


class particle():
    def __init__(self, loc, m=1.0, vel=[0.0, 0.0, 0.0] ):
        self.X=numpy.array([(loc[0]),(loc[1]),(loc[2])])
        self.V=numpy.array([(vel[0]),(vel[1]),(vel[2])])
        self.A=numpy.array([(0),(0),(0)])
        self.mass=m

class sim():
    p_store=[]

    def __init__(self,N_in , box_dims):
        self.N=N_in
        self.box_size=box_dims
        
        
        for i in range(N):
            loc=lattice_init(i,self.N, self.box_size)
            self.p_store.append(particle(loc))



    def verlet_vel(self, dt):
        for i in self.p_store:
            i.V = i.V + dt*i.A

    def verlet_pos(self, dt):
        for i in self.p_store:
            i.X = i.X + dt*i.V
            for j in range(3):
                if i.X[j] < self.box_size[j,0]:
                    i.X[j]= self.box_size[j,1] + i.X[j] - self.box_size[j,0]
                elif i.X[j] > self.box_size[j,1]:
                    i.X[j]= self.box_size[j,0] + i.X[j] - self.box_size[j,1]
            

    def accel_update(self):

        for i in self.p_store:
            i.A=numpy.array([(0),(0),(0)])

        for i in range(self.N):
            for j in range(i+1,self.N):

                #a_tmp=(self.p_store[i].X - self.p_store[j].X)
                a_tmp=numpy.subtract(self.p_store[i].X, self.p_store[j].X)
                
                r=numpy.linalg.norm(a_tmp)
                

                
                if r>0:
                    a_tmp=a_tmp*(48*((1.0/r)**(14)) - 24*((1/r)**(8)) )
                elif r <= 0:
                    print 'r = 0 error'
                

                self.p_store[i].A=numpy.add(self.p_store[i].A, a_tmp)
                self.p_store[j].A=numpy.subtract(self.p_store[j].A, a_tmp)
    
    def verlet_step(self, dt, it_max):
        for i in range(it_max):
            self.verlet_vel(dt*0.5)
            self.verlet_pos(dt)
            self.accel_update()
            self.verlet_vel(dt*0.5)
            
            
    
    def frame_plot(self):
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i in self.p_store:
                #plt.scatter(i.X[0], i.X[1], alpha=0.5)
                ax.scatter(i.X[0], i.X[1], i.X[2])
            plt.show()




if __name__ == '__main__':

    it_max=500   #max iterations
    N=64     #number of paricles
    box_size=numpy.array([(-5,5),(-5,5),(-5,5)])    #size of box containing particles
    dt=0.005

    
    s1=sim(N,box_size)
    
    
    
    
    s1.verlet_step(dt, it_max)
    s1.frame_plot()  

    

    























