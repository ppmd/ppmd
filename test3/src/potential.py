import math


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
        
class LennardJones(BasePotential):
    """
    Lennard-Jones potential:
    
    .. math:
        V(r) = 4\epsilon ((r/\sigma)^{-12} - (r/\sigma)^{-6})    
    
    
    """
    def __init__(self, epsilon = 1.0, sigma = 1.0):
        """
        Initialise Lennard Jones potential:
        
        :arg epsilon: (float) Potential parameter :math:`\epsilon`
        :arg sigma: (float) Potential parameter :math:`\sigma`   
        
        
        """
        self._epsilon = epsilon
        self._sigma = sigma
        self._sigma6 = float(sigma)**6
        self._4epsilon = 4. * self._epsilon
        self._48_sigma = 48. / self._sigma
        
        
        self._rc = 2.**(1./6.)*self._sigma
        self._rn = 2*self._rc
        self._rn2 = self._rn**2
        
        
        
    def epsilon(self):
        """
        Return :math:`\epsilon`
        """
        return self._epsilon

    def sigma(self):
        """
        Return :math:`\sigma`
        """
        return self._sigma
                
        
    def evaluate(self, r):
        """
        Evaluate potential.
        
        :arg r: (float) Inter-atomic distance :math:`r=|\\vec{r}_i-\\vec{r}_j|`
        """
        _sig6_rm6 = self._sigma6*(r**(-6))
        return _4epsilon * _sig6_rm6 * (_sig6_rm6 - 1.0)
        
    def evaluate_force(self, r):
        """
        Evaluate force.
        
        :arg r: (float) Inter-atomic distance :math:`r=|\\vec{r}_i-\\vec{r}_j|`
        
        Assumes non-dimensionalisation:
        :math: '\epsilon = 1'
        :math: '\sigma = 1'
        :math: 'mass = 1'
        
        
        """
        _rm2 = r**(-2)
        _rm6 = _rm2**(3)
        return self._48_sigma * _rm2 * _rm6 * (_rm6 - 0.5)
        
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
        self._C_F = 48*self._epsilon/self._sigma**2
        self._rc = 2.**(1./6.)*self._sigma
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







