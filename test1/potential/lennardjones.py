from . import basepotential

class LennardJones(basepotential.BasePotential):
    """
    Lennard-Jones potential:
    
    .. math:
        V(r) = 4\epsilon ((r/\sigma)^{-12} - (r/\sigma)^{-6})    
    

    :arg epsilon: Potential parameter :math:`\epsilon`
    :arg sigma: Potential parameter :math:`\sigma`    
    
    """
    def __init__(self, epsilon = 1.0, sigma = 1.0):
        self._epsilon = epsilon
        self._sigma = sigma
        self._sigma6 = float(sigma)**6
        self._4epsilon = 4. * self._epsilon
        self._48_sigma = 48. / self._sigma
        
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
        
        :arg r: Inter-atomic distance :math:`r=|\\vec{r}_i-\\vec{r}_j|`
        """
        _sig6_rm6 = self._sigma6*(r**(-6))
        return _4epsilon * _sig6_rm6 * (_sig6_rm6 - 1.0)
        
    def evaluate_force(self, r):
        """
        Evaluate force.
        
        :arg r: Inter-atomic distance :math:`r=|\\vec{r}_i-\\vec{r}_j|`
        
        Assumes non-dimensionalisation:
        :math: '\epsilon = 1'
        :math: '\sigma = 1'
        :math: 'mass = 1'
        
        
        """
        _rm2 = r**(-2)
        _rm6 = _rm2**(3)
        return self._48_sigma * _rm2 * _rm6 * (_rm6 - 0.5)
