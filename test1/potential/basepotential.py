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
