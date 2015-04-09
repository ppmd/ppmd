import math
import kernel
import constant


class BasePotential(object):
    '''Abstract base class for inter-atomic potentials.

    Inter-atomic potentials can be described by a scalar
    function :math:`V(r)` which depends on the distance :math:`r=|\\vec{r}_i-\\vec{r}_j|`
    between two atoms i and j. In MD simulations we need this potential and
    the force :math:`\\vec{F}(r) = -\\nabla V(r)`.
    '''
    def __init__(self):
        pass

    def evaluate(self,r):
        '''Evaluate potential :math:`V(r)`
    
        :arg r: Inter-atom distance :math:`r=|r_i-r_j|`
        '''

    def evaluate_force(self,rx,ry):
        '''Evaluate force.

        Calculate the interatomic force :math:`\\vec{F}(r) = -\\nabla V(r)` for
        atomic distance :\\vec{r}=(r_x,r_y)=\\vec{r}_i - \\vec{r}_j:

        :arg rx: x-component of distance :math:`r_x`
        :arg ry: y-component of distance :math:`r_y`
        '''
        
################################################################################################################
# LJ 2**(1/6) sigma
################################################################################################################          
        
class LennardJonesShifted(BasePotential):
    '''Shifted Lennard Jones potential.
    
    .. math:
        V(r) = 4\epsilon ((r/\sigma)^{-6} - (r/\sigma)^{-12} + 1/4)
        
    for :math:`r>r_c=2^{1/6}` the potential (and force) is set to zero.

    :arg epsilon: Potential parameter :math:`\epsilon`
    :arg sigma: Potential parameter :math:`\sigma`
    '''
    def __init__(self,epsilon = 1.0,sigma = 1.0):
        self._epsilon = epsilon
        self._sigma = sigma
        self._C_V = 4.*self._epsilon
        self._C_F = -48*self._epsilon/self._sigma**2
        self._rc = 2.**(1./6.)*self._sigma
        self._rn = 1.2*self._rc
        self._rc2 = self._rc**2
        self._sigma2 = self._sigma**2
        

    @property
    def epsilon(self):
        '''Value of parameter :math:`\epsilon`'''
        return self._epsilon

    @property
    def sigma(self):
        '''Value of parameter :math:`\sigma`'''
        return self._sigma

    @property
    def rc(self):
        '''Value of cufoff distance :math:`r_c`'''
        return self._rc

    def evaluate(self,r):
        '''Evaluate potential.

        :arg r: Inter-atomic distance :math:`r=|\\vec{r}_i-\\vec{r}_j|`
        '''
        if (r < self._rc):
            r_m6 = (r/self._sigma)**(-6)
            return self._C_V*((r_m6-1.0)*r_m6 + 0.25)
        else:
            return 0.0
    """
    def evaluate_force(self,rx,ry):
        '''Evaluate force.

        :arg rx: x-component of distance :math:`r_x`
        :arg ry: y-component of distance :math:`r_y`
        '''
        r2 = rx**2+ry**2
        if (r2 < self._rc2):
            r_m2 = self._sigma2/r2
            tmp = self._C_F*(r_m2**7 - 0.5*r_m2**4)
            return (tmp*rx,tmp*ry)
        else:
            return (0.0,0.0)
    """
    def evaluate_force(self,r2):
        '''Evaluate force.

        :arg rx: x-component of distance :math:`r_x`
        :arg ry: y-component of distance :math:`r_y`
        '''
        
        if (r2 < self._rc2):
            r_m2 = self._sigma2/r2
            return  self._C_F*(r_m2**7 - 0.5*r_m2**4)
            
        else:
            return 0.0   
    
    
    def kernel(self):
        '''
        Returns a kernel class for the potential.
        '''
        
        kernel_code = '''
        
        const double R0 = P[1][0] - P[0][0];
        const double R1 = P[1][1] - P[0][1];
        const double R2 = P[1][2] - P[0][2];
        
        const double r2 = R0*R0 + R1*R1 + R2*R2;
        
        if (r2 < rc2){
        

            /* Lennard-Jones */
            const double r_m2 = sigma2/r2;
            const double r_m4 = r_m2*r_m2;
            const double r_m6 = r_m4*r_m2;
            
            *U+= CV*((r_m6-1.0)*r_m6 + 0.25);
            
            const double r_m8 = r_m4*r_m4;
            const double f_tmp = CF*(r_m6 - 0.5)*r_m8;

            
            A[0][0]+=f_tmp*R0;
            A[0][1]+=f_tmp*R1;
            A[0][2]+=f_tmp*R2;
            
            A[1][0]-=f_tmp*R0;
            A[1][1]-=f_tmp*R1;
            A[1][2]-=f_tmp*R2;

        }
        
        '''
        constants=(constant.Constant('sigma2',self._sigma**2),
                   constant.Constant('rc2',self._rc**2),
                   constant.Constant('CF',self._C_F),
                   constant.Constant('CV',self._C_V))        
        
        
        return kernel.Kernel('LJ_accel_U',kernel_code, constants)

    def datdict(self, input_state):
        '''
        Map between state variables and kernel variables, returns required dictonary.
        
        :arg state input_state: state with containing variables.
        '''
        return {'P':input_state.positions, 'A':input_state.accelerations, 'U':input_state.U}
        
               
        
################################################################################################################
# LJ 2.5 sigma
################################################################################################################  
        
class LennardJones(LennardJonesShifted):
    '''Lennard Jones potential.
    
    .. math:
        V(r) = 4\epsilon ((r/\sigma)^{-6} - (r/\sigma)^{-12} + U(5/2 \sigma))
        
    for :math:`r>r_c=(5/2) \sigma` the potential (and force) is set to zero.

    :arg epsilon: Potential parameter :math:`\epsilon`
    :arg sigma: Potential parameter :math:`\sigma`
    '''
    def __init__(self,epsilon = 1.0,sigma = 1.0):
        self._epsilon = epsilon
        self._sigma = sigma
        self._C_V = 4.*self._epsilon
        self._C_F = -48*self._epsilon/self._sigma**2
        self._rc = self._sigma*(5./2.)
        self._rn = 1.2*self._rc
        self._rc2 = self._rc**2
        self._sigma2 = self._sigma**2
        self._shift_internal = (2.0/5.0)**6 - (2.0/5.0)**12
        
    def evaluate(self,r):
        '''Evaluate potential.

        :arg r: Inter-atomic distance :math:`r=|\\vec{r}_i-\\vec{r}_j|`
        '''
        if (r < self._rc):
            r_m6 = (r/self._sigma)**(-6)
            return self._C_V*((r_m6-1.0)*r_m6 + self._shift_internal)
        else:
            return 0.0        
        
    def kernel(self):
        '''
        Returns a kernel class for the potential.
        '''
        
        kernel_code = '''
        
        const double R0 = P[1][0] - P[0][0];
        const double R1 = P[1][1] - P[0][1];
        const double R2 = P[1][2] - P[0][2];
        
        const double r2 = R0*R0 + R1*R1 + R2*R2;
        
        if (r2 < rc2){
        

            /* Lennard-Jones */
            const double r_m2 = sigma2/r2;
            const double r_m4 = r_m2*r_m2;
            const double r_m6 = r_m4*r_m2;
            
            *U+= CV*((r_m6-1.0)*r_m6 + internalshift);
            
            const double r_m8 = r_m4*r_m4;
            const double f_tmp = CF*(r_m6 - 0.5)*r_m8;

            
            A[0][0]+=f_tmp*R0;
            A[0][1]+=f_tmp*R1;
            A[0][2]+=f_tmp*R2;
            
            A[1][0]-=f_tmp*R0;
            A[1][1]-=f_tmp*R1;
            A[1][2]-=f_tmp*R2;

        }
        
        '''
        constants=(constant.Constant('sigma2',self._sigma**2),
                   constant.Constant('rc2',self._rc**2),
                   constant.Constant('internalshift',self._shift_internal),
                   constant.Constant('CF',self._C_F),
                   constant.Constant('CV',self._C_V))        
        
        
        return kernel.Kernel('LJ_accel_U',kernel_code, constants)





























