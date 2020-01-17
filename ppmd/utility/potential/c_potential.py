__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"


# system level
import math
import ctypes

# package level
from ppmd import kernel, data, access


class BasePotential(object):
    r"""Abstract base class for inter-atomic potentials.

    Inter-atomic potentials can be described by a scalar
    function :math:`V(r)` which depends on the distance :math:`r=|\\vec{r}_i-\\vec{r}_j|`
    between two atoms i and j. In MD simulations we need this potential and
    the force :math:`\\vec{F}(r) = -\\nabla V(r)`.
    """

    def __init__(self):
        pass


################################################################################################################
# LJ 2**(1/6) sigma
################################################################################################################          

class LennardJonesShifted(BasePotential):
    r"""Shifted Lennard Jones potential.
    
    .. math:
        V(r) = 4\epsilon ((r/\sigma)^{-6} - (r/\sigma)^{-12} + 1/4)
        
    for :math:`r>r_c=2^{1/6}` the potential (and force) is set to zero.

    :arg epsilon: Potential parameter :math:`\epsilon`
    :arg sigma: Potential parameter :math:`\sigma`
    """

    def __init__(self, epsilon=1.0, sigma=1.0):
        self._epsilon = epsilon
        self._sigma = sigma
        self._C_V = 4. * self._epsilon
        self._C_F = -48 * self._epsilon / self._sigma ** 2
        self._rc = 2. ** (1. / 6.) * self._sigma
        self._rn = 1.2 * self._rc
        self._rc2 = self._rc ** 2
        self._sigma2 = self._sigma ** 2

    @property
    def rc(self):
        """Value of cufoff distance :math:`r_c`"""
        return self._rc

    @property
    def kernel(self):
        """
        Returns a kernel class for the potential.
        """

        kernel_code = '''

        const double R0 = P(1, 0) - P(0, 0);
        const double R1 = P(1, 1) - P(0, 1);
        const double R2 = P(1, 2) - P(0, 2);

        const double r2 = R0*R0 + R1*R1 + R2*R2;

        if (r2 < rc2){

            const double r_m2 = sigma2/r2;
            const double r_m4 = r_m2*r_m2;
            const double r_m6 = r_m4*r_m2;

            u(0)+= CV*((r_m6-1.0)*r_m6 + 0.25);

            const double r_m8 = r_m4*r_m4;
            const double f_tmp = CF*(r_m6 - 0.5)*r_m8;

            A(0, 0)+=f_tmp*R0;
            A(0, 1)+=f_tmp*R1;
            A(0, 2)+=f_tmp*R2;

            A(1, 0)-=f_tmp*R0;
            A(1, 1)-=f_tmp*R1;
            A(1, 2)-=f_tmp*R2;

            }

        '''
        constants = (kernel.Constant('sigma2', self._sigma ** 2),
                     kernel.Constant('rc2', self._rc ** 2),
                     kernel.Constant('internalshift', self._shift_internal),
                     kernel.Constant('CF', self._C_F),
                     kernel.Constant('CV', self._C_V))

        return kernel.Kernel('LJ_accel_U', kernel_code, constants, [kernel.Header('stdio.h')])

    def datdict(self, input_state):
        """
        Map between state variables and kernel variables, returns required dictonary.
        
        :arg state input_state: state with containing variables.
        """

        return {'P': input_state.positions(access.R), 'A': input_state.forces(access.INC0), 'u': input_state.u(access.INC)}

    def get_data_map(self, positions=None, forces=None, potential_energy=None):
         return {'P': positions(access.R), 'A': forces(access.INC0), 'u': potential_energy(access.INC0)}

################################################################################################################
# LJ 2.5 sigma
################################################################################################################  

class LennardJones(LennardJonesShifted):
    r"""Lennard Jones potential.
    
    .. math:
        V(r) = 4\epsilon ((r/\sigma)^{-6} - (r/\sigma)^{-12} + u(5/2 \sigma))
        
    for :math:`r>r_c=(5/2) \sigma` the potential (and force) is set to zero.

    :arg epsilon: Potential parameter :math:`\epsilon`
    :arg sigma: Potential parameter :math:`\sigma`
    """

    def __init__(self, epsilon=1.0, sigma=1.0, rc=None):
        self._epsilon = epsilon
        self._sigma = sigma
        self._C_V = 4. * self._epsilon
        self._C_F = -48 * self._epsilon / self._sigma ** 2
        if rc is None:
            self._rc = self._sigma * (5. / 2.)
        else:
            self._rc = rc

        self._rn = 1. * self._rc
        self._rc2 = self._rc ** 2
        self._sigma2 = self._sigma ** 2
        self._shift_internal = (self._sigma / self._rc) ** 6 - (self._sigma / self._rc) ** 12


    @property
    def kernel(self):
        """
        Returns a kernel class for the potential.
        """

        kernel_code = '''

        const double R0 = P(1, 0) - P(0, 0);
        const double R1 = P(1, 1) - P(0, 1);
        const double R2 = P(1, 2) - P(0, 2);

        const double r2 = R0*R0 + R1*R1 + R2*R2;

        //printf("r2: %f\\n", r2);
        if (r2 < rc2){

            const double r_m2 = sigma2/r2;
            const double r_m4 = r_m2*r_m2;
            const double r_m6 = r_m4*r_m2;

            u(0)+= CV*((r_m6-1.0)*r_m6 + internalshift);

            const double r_m8 = r_m4*r_m4;
            const double f_tmp = CF*(r_m6 - 0.5)*r_m8;

            A(0, 0)+=f_tmp*R0;
            A(0, 1)+=f_tmp*R1;
            A(0, 2)+=f_tmp*R2;

            A(1, 0)-=f_tmp*R0;
            A(1, 1)-=f_tmp*R1;
            A(1, 2)-=f_tmp*R2;

            //printf("ftmp: %f\\n",f_tmp);

            }

        '''
        constants = (kernel.Constant('sigma2', self._sigma ** 2),
                     kernel.Constant('rc2', self._rc ** 2),
                     kernel.Constant('internalshift', self._shift_internal),
                     kernel.Constant('CF', self._C_F),
                     kernel.Constant('CV', self._C_V))

        return kernel.Kernel('LJ_accel_U',
                             kernel_code,
                             constants,
                             [kernel.Header('stdio.h')])





################################################################################################################
# NULL Potential, returns zero for accel
################################################################################################################  

class NULL(object):
    """
    
    NULL potential
    
    """

    def __init__(self, rc=None):
        self._rc = rc
        self._rn = rc

    @property
    def rc(self):
        """Value of cufoff distance :math:`r_c`"""
        return self._rc

    @property
    def rn(self):
        """Value of cufoff distance :math:`r_c`"""
        return self._rc

    @property
    def kernel(self):
        """
        Returns a kernel class for the potential.
        """

        kernel_code = '''
        
        const double R0 = P(1, 0) - P(0, 0);
        const double R1 = P(1, 1) - P(0, 1);
        const double R2 = P(1, 2) - P(0, 2);
        
        A(0, 0)=0;
        A(0, 1)=0;
        A(0, 2)=0;
        
        A(1, 0)=0;
        A(1, 1)=0;
        A(1, 2)=0;
        
        '''

        return kernel.Kernel('NULL_Potential', kernel_code, None, None, None)

    @staticmethod
    def datdict(input_state):
        """
        Map between state variables and kernel variables, returns required dictonary.
        
        :arg state input_state: state with containing variables.
        """
        return {'P': input_state.positions(access.R), 'A': input_state.forces(access.INC0), 'u': input_state.u(access.INC0)}


class LennardJonesCounter(LennardJones):
    r"""Lennard Jones potential.
    
    .. math:
        V(r) = 4 \epsilon ((r/\sigma)^{-6} - (r/\sigma)^{-12} + u(5/2 \sigma))
        
    for :math:`r>r_c=(5/2) \sigma` the potential (and force) is set to zero.

    :arg epsilon: Potential parameter :math:`\epsilon`
    :arg sigma: Potential parameter :math:`\sigma`
    """

    def __init__(self, epsilon=1.0, sigma=1.0, rc=None):
        self._epsilon = epsilon
        self._sigma = sigma
        self._C_V = 4. * self._epsilon
        self._C_F = -48 * self._epsilon / self._sigma ** 2
        if rc is None:
            self._rc = self._sigma * (5. / 2.)
        else:
            self._rc = rc

        self._rn = 1. * self._rc
        self._rc2 = self._rc ** 2
        self._sigma2 = self._sigma ** 2
        self._shift_internal = (self._sigma / self._rc) ** 6 - (self._sigma / self._rc) ** 12

        self._counter = data.ScalarArray([0], dtype=ctypes.c_longlong, name="counter")
        self._counter.data[0] = 0
        self._counter_outer = data.ScalarArray([0], dtype=ctypes.c_longlong, name="counter")
        self._counter_outer.data[0] = 0

    @property
    def kernel(self):
        """
        Returns a kernel class for the potential.
        """

        kernel_code = '''

        OUTCOUNT(0)++;

        const double R0 = P(1, 0) - P(0, 0);
        const double R1 = P(1, 1) - P(0, 1);
        const double R2 = P(1, 2) - P(0, 2);


        //printf("Positions P(0) = %f, P(1) = %f |", P(0, 1), P(1, 1));


        const double r2 = R0*R0 + R1*R1 + R2*R2;

        if (r2 < rc2){

            COUNT(0)++;

            const double r_m2 = sigma2/r2;
            const double r_m4 = r_m2*r_m2;
            const double r_m6 = r_m4*r_m2;

            u(0)+= CV*((r_m6-1.0)*r_m6 + internalshift);

            const double r_m8 = r_m4*r_m4;
            const double f_tmp = CF*(r_m6 - 0.5)*r_m8;


            A(0, 0)+=f_tmp*R0;
            A(0, 1)+=f_tmp*R1;
            A(0, 2)+=f_tmp*R2;

            A(1, 0)-=f_tmp*R0;
            A(1, 1)-=f_tmp*R1;
            A(1, 2)-=f_tmp*R2;

        }

        '''
        constants = (kernel.Constant('sigma2', self._sigma ** 2),
                     kernel.Constant('rc2', self._rc ** 2),
                     kernel.Constant('internalshift', self._shift_internal),
                     kernel.Constant('CF', self._C_F),
                     kernel.Constant('CV', self._C_V))

        reductions = (kernel.Reduction('u', 'u[0]', '+'),)

        return kernel.Kernel('LJ_accel_U', kernel_code, constants, [kernel.Header('stdio.h')], reductions)

    def datdict(self, input_state):
        """
        Map between state variables and kernel variables, returns required dictonary.
        
        :arg state input_state: state with containing variables.
        """
        return {'P': input_state.positions(access.R), 'A': input_state.forces(access.INC0), 'u': input_state.u(access.INC0), 'COUNT': self._counter(access.INC),
                'OUTCOUNT': self._counter_outer(access.INC)}

class TestPotential1(LennardJones):
    r"""Lennard Jones potential.

    .. math:
        V(r) = 4\epsilon ((r/\sigma)^{-6} - (r/\sigma)^{-12} + u(5/2 \sigma))

    for :math:`r>r_c=(5/2) \sigma` the potential (and force) is set to zero.

    :arg epsilon: Potential parameter :math:`\epsilon`
    :arg sigma: Potential parameter :math:`\sigma`
    """



    @property
    def kernel(self):
        """
        Returns a kernel class for the potential.
        """

        kernel_code = '''

        const double R[3] = {P[1][0] - P[0][0], P[1][1] - P[0][1], P[1][2] - P[0][2]};

        double r2 = R[0]*R[0] + R[1]*R[1] + R[2]*R[2];

        if (r2 < rc2){

            r2=1./r2;

            A[0][0]+=r2;
            A[0][1]+=r2;
            A[0][2]+=r2;

            A[1][0]+=r2;
            A[1][1]+=r2;
            A[1][2]+=r2;

        }
        '''
        constants = (kernel.Constant('rc2', self._rc ** 2),)

        return kernel.Kernel('TestPotential1', kernel_code, constants, ['stdio.h'], None)


class TestPotential2(LennardJones):
    r"""Lennard Jones potential.

    .. math:
        V(r) = 4\epsilon ((r/\sigma)^{-6} - (r/\sigma)^{-12} + u(5/2 \sigma))

    for :math:`r>r_c=(5/2) \sigma` the potential (and force) is set to zero.

    :arg epsilon: Potential parameter :math:`\epsilon`
    :arg sigma: Potential parameter :math:`\sigma`
    """


    @property
    def kernel(self):
        """
        Returns a kernel class for the potential.
        """

        kernel_code = '''

        const double R0 = P(1, 0) - P(0, 0);
        const double R1 = P(1, 1) - P(0, 1);
        const double R2 = P(1, 2) - P(0, 2);

        const double r2 = R0*R0 + R1*R1 + R2*R2;

            const double r_m2 = sigma2/r2;
            const double r_m4 = r_m2*r_m2;
            const double r_m6 = r_m4*r_m2;

            u(0)+= CV*((r_m6-1.0)*r_m6 + internalshift);

            const double r_m8 = r_m4*r_m4;
            const double f_tmp = CF*(r_m6 - 0.5)*r_m8;

            A(0, 0)+=f_tmp*R0;
            A(0, 1)+=f_tmp*R1;
            A(0, 2)+=f_tmp*R2;

            A(1, 0)-=f_tmp*R0;
            A(1, 1)-=f_tmp*R1;
            A(1, 2)-=f_tmp*R2;

        '''
        constants = (kernel.Constant('sigma2', self._sigma ** 2),
                     kernel.Constant('rc2', self._rc ** 2),
                     kernel.Constant('internalshift', self._shift_internal),
                     kernel.Constant('CF', self._C_F),
                     kernel.Constant('CV', self._C_V))

        reductions = (kernel.Reduction('u', 'u[0]', '+'),)

        return kernel.Kernel('LJ_accel_U', kernel_code, constants, ['stdio.h'], reductions)





class TestPotential3(LennardJones):
    r"""Lennard Jones potential.

    .. math:
        V(r) = 4\epsilon ((r/\sigma)^{-6} - (r/\sigma)^{-12} + u(5/2 \sigma))

    for :math:`r>r_c=(5/2) \sigma` the potential (and force) is set to zero.

    :arg epsilon: Potential parameter :math:`\epsilon`
    :arg sigma: Potential parameter :math:`\sigma`
    """


    @property
    def kernel(self):
        """
        Returns a kernel class for the potential.
        """

        kernel_code = '''

        const double R0 = P(1, 0) - P(0, 0);
        const double R1 = P(1, 1) - P(0, 1);
        const double R2 = P(1, 2) - P(0, 2);

        const double r2 = R0*R0 + R1*R1 + R2*R2;

            const double r_m2 = sigma2/r2;
            const double r_m4 = r_m2*r_m2;
            const double r_m6 = r_m4*r_m2;


            u(0)+=(r2 < rc2) ? CV*((r_m6-1.0)*r_m6 + internalshift) : 0.0 ;

            const double r_m8 = r_m4*r_m4;
            const double f_tmp = CF*(r_m6 - 0.5)*r_m8;

            A(0, 0)+=(r2 < rc2) ? f_tmp*R0 : 0.0;
            A(0, 1)+=(r2 < rc2) ? f_tmp*R1 : 0.0;
            A(0, 2)+=(r2 < rc2) ? f_tmp*R2 : 0.0;

            A(1, 0)-=(r2 < rc2) ? f_tmp*R0 : 0.0;
            A(1, 1)-=(r2 < rc2) ? f_tmp*R1 : 0.0;
            A(1, 2)-=(r2 < rc2) ? f_tmp*R2 : 0.0;


        '''
        constants = (kernel.Constant('sigma2', self._sigma ** 2),
                     kernel.Constant('rc2', self._rc ** 2),
                     kernel.Constant('internalshift', self._shift_internal),
                     kernel.Constant('CF', self._C_F),
                     kernel.Constant('CV', self._C_V))

        reductions = (kernel.Reduction('u', 'u[0]', '+'),)

        return kernel.Kernel('LJ_accel_U', kernel_code, constants, ['stdio.h'], reductions)


class TestPotential4(LennardJones):
    r"""Lennard Jones potential.

    .. math:
        V(r) = 4\epsilon ((r/\sigma)^{-6} - (r/\sigma)^{-12} + u(5/2 \sigma))

    for :math:`r>r_c=(5/2) \sigma` the potential (and force) is set to zero.

    :arg epsilon: Potential parameter :math:`\epsilon`
    :arg sigma: Potential parameter :math:`\sigma`
    """


    @property
    def kernel(self):
        """
        Returns a kernel class for the potential.
        """

        kernel_code = '''

        const double R0 = P(1, 0) - P(0, 0);
        const double R1 = P(1, 1) - P(0, 1);
        const double R2 = P(1, 2) - P(0, 2);

        const double r2 = R0*R0 + R1*R1 + R2*R2;

            double xn = 0.01;
            for(int ix = 0; ix < 10; ix++){
                xn = xn*(2.0 - r2*xn);
            }



            const double r_m2 = sigma2*xn;
            const double r_m4 = r_m2*r_m2;
            const double r_m6 = r_m4*r_m2;

            const double _ex = r_m6;
            double _et = 1.0, _ep = 1.0, _ef = 1.0, _epx = 1.0;
            for(int _etx = 1; _etx < 21; _etx++){
                _epx *= _ex;
                _ef *= _ep;
                _ep++;

                xn = 0.01;
                for(int ix = 0; ix < 10; ix++){
                    xn = xn*(2.0 - _ef*xn);
                }


                _et += _epx*xn;
            }

            u(0)+=CV*((r_m6-1.0)*r_m6 + internalshift) + _et;

            const double r_m8 = r_m4*r_m4;
            const double f_tmp = CF*(r_m6 - 0.5)*r_m8;

            A(0, 0)+=f_tmp*R0;
            A(0, 1)+=f_tmp*R1;
            A(0, 2)+=f_tmp*R2;

            A(1, 0)-=f_tmp*R0;
            A(1, 1)-=f_tmp*R1;
            A(1, 2)-=f_tmp*R2;


        '''
        constants = (kernel.Constant('sigma2', self._sigma ** 2),
                     kernel.Constant('rc2', self._rc ** 2),
                     kernel.Constant('internalshift', self._shift_internal),
                     kernel.Constant('CF', self._C_F),
                     kernel.Constant('CV', self._C_V))

        reductions = (kernel.Reduction('u', 'u[0]', '+'),)

        return kernel.Kernel('LJ_accel_U', kernel_code, constants, ['stdio.h'], reductions)


class TestPotential4p(LennardJones):
    r"""Lennard Jones potential.

    .. math:
        V(r) = 4\epsilon ((r/\sigma)^{-6} - (r/\sigma)^{-12} + u(5/2 \sigma))

    for :math:`r>r_c=(5/2) \sigma` the potential (and force) is set to zero.

    :arg epsilon: Potential parameter :math:`\epsilon`
    :arg sigma: Potential parameter :math:`\sigma`
    """


    @property
    def kernel(self):
        """
        Returns a kernel class for the potential.
        """

        kernel_code = '''

        const double R0 = P[1][0] - P[0][0];
        const double R1 = P[1][1] - P[0][1];
        const double R2 = P[1][2] - P[0][2];

        const double r2 = R0*R0 + R1*R1 + R2*R2;


            double xn = 0.01;
            for(int ix = 0; ix < 2; ix++){
                xn = xn*(2.0 - r2*xn);
            }


            const double r_m2 = sigma2*xn;
            const double r_m4 = r_m2*r_m2;
            const double r_m6 = r_m4*r_m2;

            const double _ex = r_m6;
            double _et = 1.0, _ep = 1.0, _ef = 1.0, _epx = 1.0;

            /*
            #pragma novector
            for(int _etx = 1; _etx < 21; _etx++){
                _epx *= _ex;
                _ef *= _ep;
                _ep++;

                xn = 0.01;

            #pragma novector
                for(int ix = 0; ix < 10; ix++){
                    xn = xn*(2.0 - _ef*xn);
                }


                _et += _epx*xn;
            }
            */

            u[0]+=CV*((r_m6-1.0)*r_m6 + internalshift);

            const double r_m8 = r_m4*r_m4;
            const double f_tmp = CF*(r_m6 - 0.5)*r_m8;

            A[0][0]+=f_tmp*R0;
            A[0][1]+=f_tmp*R1;
            A[0][2]+=f_tmp*R2;

            A[1][0]-=f_tmp*R0;
            A[1][1]-=f_tmp*R1;
            A[1][2]-=f_tmp*R2;


        '''
        constants = (kernel.Constant('sigma2', self._sigma ** 2),
                     kernel.Constant('rc2', self._rc ** 2),
                     kernel.Constant('internalshift', self._shift_internal),
                     kernel.Constant('CF', self._C_F),
                     kernel.Constant('CV', self._C_V))

        reductions = (kernel.Reduction('u', 'u[0]', '+'),)

        return kernel.Kernel('LJ_accel_U',
                             kernel_code,
                             constants,
                             [kernel.Header('stdio.h')],
                             reductions)

class VLennardJones(LennardJones):
    @property
    def kernel(self):
        """
        Returns a kernel class for the potential.
        """

        kernel_code = '''
        //N_f = 27
        const double R0 = P.j[0] - P.i[0];
        const double R1 = P.j[1] - P.i[1];
        const double R2 = P.j[2] - P.i[2];

        const double r2 = R0*R0 + R1*R1 + R2*R2;

        const double r_m2 = sigma2/r2;
        const double r_m4 = r_m2*r_m2;
        const double r_m6 = r_m4*r_m2;

        u[0]+= (r2 < rc2) ? 0.5*CV*((r_m6-1.0)*r_m6 + internalshift) : 0.0;

        const double r_m8 = r_m4*r_m4;
        const double f_tmp = CF*(r_m6 - 0.5)*r_m8;

        A.i[0]+= (r2 < rc2) ? f_tmp*R0 : 0.0;
        A.i[1]+= (r2 < rc2) ? f_tmp*R1 : 0.0;
        A.i[2]+= (r2 < rc2) ? f_tmp*R2 : 0.0;
        '''
        constants = (kernel.Constant('sigma2', self._sigma ** 2),
                     kernel.Constant('rc2', self._rc ** 2),
                     kernel.Constant('internalshift', self._shift_internal),
                     kernel.Constant('CF', self._C_F),
                     kernel.Constant('CV', self._C_V))

        return kernel.Kernel('LJ_accel_U',
                             kernel_code,
                             constants,
                             [kernel.Header('stdio.h')])

class NoVLennardJones(LennardJones):
    @property
    def kernel(self):
        """
        Returns a kernel class for the potential.
        """

        kernel_code = '''
        //N_f = 27
        const double R0 = P.j[0] - P.i[0];
        const double R1 = P.j[1] - P.i[1];
        const double R2 = P.j[2] - P.i[2];

        const double r2 = R0*R0 + R1*R1 + R2*R2;

        if (r2 < rc2){

            const double r_m2 = sigma2/r2;
            const double r_m4 = r_m2*r_m2;
            const double r_m6 = r_m4*r_m2;

            u[0] += 0.5*CV*((r_m6-1.0)*r_m6 + internalshift);

            const double r_m8 = r_m4*r_m4;
            const double f_tmp = CF*(r_m6 - 0.5)*r_m8;

            A.i[0] +=  f_tmp*R0;
            A.i[1] +=  f_tmp*R1;
            A.i[2] +=  f_tmp*R2;

        }
        '''
        constants = (kernel.Constant('sigma2', self._sigma ** 2),
                     kernel.Constant('rc2', self._rc ** 2),
                     kernel.Constant('internalshift', self._shift_internal),
                     kernel.Constant('CF', self._C_F),
                     kernel.Constant('CV', self._C_V))

        return kernel.Kernel('LJ_accel_U',
                             kernel_code,
                             constants,
                             [kernel.Header('stdio.h')])

class VLennardJonesNoU(LennardJones):
    @property
    def kernel(self):
        """
        Returns a kernel class for the potential.
        """

        kernel_code = '''

        const double R0 = P.j[0] - P.i[0];
        const double R1 = P.j[1] - P.i[1];
        const double R2 = P.j[2] - P.i[2];

        const double r2 = R0*R0 + R1*R1 + R2*R2;

        const double r_m2 = sigma2/r2;
        const double r_m4 = r_m2*r_m2;

        const double f_tmp = CF*(r_m4*r_m2 - 0.5)*r_m4*r_m4;

        A.i[0]+= (r2 < rc2) ? f_tmp*R0 : 0.0;
        A.i[1]+= (r2 < rc2) ? f_tmp*R1 : 0.0;
        A.i[2]+= (r2 < rc2) ? f_tmp*R2 : 0.0;

        '''
        constants = (kernel.Constant('sigma2', self._sigma ** 2),
                     kernel.Constant('rc2', self._rc ** 2),
                     kernel.Constant('internalshift', self._shift_internal),
                     kernel.Constant('CF', self._C_F),
                     kernel.Constant('CV', self._C_V))

        return kernel.Kernel('LJ_accel',
                             kernel_code,
                             constants,
                             [kernel.Header('stdio.h')])

    def get_data_map(self, positions=None, forces=None, potential_energy=None):
        if potential_energy is not None:
            print("warning, kernel does not compute potential energy")
        return {'P': positions(access.R), 'A': forces(access.INC0)}


class VLennardJones2(LennardJones):
    @property
    def kernel(self):
        """
        Returns a kernel class for the potential.
        """

        kernel_code = '''

        const double R0 = P(1, 0) - P(0, 0);
        const double R1 = P(1, 1) - P(0, 1);
        const double R2 = P(1, 2) - P(0, 2);

        const double r2 = R0*R0 + R1*R1 + R2*R2;

        const double r_m2 = sigma2/r2;
        const double r_m4 = r_m2*r_m2;
        const double r_m6 = r_m4*r_m2;

        u(0)+= (r2 < rc2) ? 0.5*CV*((r_m6-1.0)*r_m6 + internalshift) : 0.0;

        const double r_m8 = r_m4*r_m4;
        const double f_tmp = CF*(r_m6 - 0.5)*r_m8;

        A(0, 0)+= (r2 < rc2) ? f_tmp*R0 : 0.0;
        A(0, 1)+= (r2 < rc2) ? f_tmp*R1 : 0.0;
        A(0, 2)+= (r2 < rc2) ? f_tmp*R2 : 0.0;

        '''
        constants = (kernel.Constant('sigma2', self._sigma ** 2),
                     kernel.Constant('rc2', self._rc ** 2),
                     kernel.Constant('internalshift', self._shift_internal),
                     kernel.Constant('CF', self._C_F),
                     kernel.Constant('CV', self._C_V))

        return kernel.Kernel('LJ_accel_U',
                             kernel_code,
                             constants, [kernel.Header('stdio.h')])



class Buckingham(BasePotential):
    """
    """

    def __init__(self, a=1.0, b=1.0, c=1.0, rc=2.5):


        self.a = a
        self.b = b
        self.mb = -1.0 * b
        self.c = c
        self.ab = self.a * self.b

        self.rc = rc
        self._shift_internal = -1.0 * a * math.exp(b*(-1.0/rc)) + c*(-1.0/(rc**6.0))

        self.rn = 1.2 * self.rc
        self.rc2 = self.rc ** 2


    @property
    def kernel(self):
        """
        Returns a kernel class for the potential.
        """

        kernel_code = '''

        const double R0 = P.j[0] - P.i[0];
        const double R1 = P.j[1] - P.i[1];
        const double R2 = P.j[2] - P.i[2];

        const double r2 = R0*R0 + R1*R1 + R2*R2;

        const double r = sqrt(r2);
        // \\exp{-B*r}
        const double exp_mbr = exp(_MB*r);

        // r^{-2, -4, -6}
        const double r_m1 = 1.0/r;
        const double r_m2 = r_m1*r_m1;
        const double r_m4 = r_m2*r_m2;
        const double r_m6 = r_m4*r_m2;

        // \\frac{C}{r^6}
        const double crm6 = _C*r_m6;

        // A \\exp{-Br} - \\frac{C}{r^6}
        u[0]+= (r2 < rc2) ? 0.5*(_A*exp_mbr - crm6 + internalshift) : 0.0;

        // = AB \\exp{-Br} - \\frac{C}{r^6}*\\frac{6}{r}
        const double term2 = crm6*(-6.0)*r_m1;
        const double f_tmp = _AB * exp_mbr + term2;

        A.i[0]+= (r2 < rc2) ? f_tmp*R0 : 0.0;
        A.i[1]+= (r2 < rc2) ? f_tmp*R1 : 0.0;
        A.i[2]+= (r2 < rc2) ? f_tmp*R2 : 0.0;

        '''
        constants = (
                     kernel.Constant('_A', self.a),
                     kernel.Constant('_AB', self.ab),
                     kernel.Constant('_B', self.b),
                     kernel.Constant('_MB', self.mb),
                     kernel.Constant('_C', self.c),
                     kernel.Constant('rc2', self.rc ** 2),
                     kernel.Constant('internalshift', self._shift_internal)
                     )

        return kernel.Kernel('BuckinghamV',
                             kernel_code,
                             constants,
                             [kernel.Header('stdio.h'), kernel.Header('math.h')])

    def datdict(self, input_state):
        """
        Map between state variables and kernel variables, returns required dictonary.

        :arg state input_state: state with containing variables.
        """

        return {'P': input_state.positions(access.R), 'A': input_state.forces(access.INC0), 'u': input_state.u(access.INC0)}

    def get_data_map(self, positions=None, forces=None, potential_energy=None):
         return {'P': positions(access.R), 'A': forces(access.INC0), 'u': potential_energy(access.INC0)}

class BuckinghamSymmetric(Buckingham):
    @property
    def kernel(self):
        """
        Returns a kernel class for the potential.
        """

        kernel_code = '''

        const double R0 = P(1,0) - P(0,0);
        const double R1 = P(1,1) - P(0,1);
        const double R2 = P(1,2) - P(0,2);

        const double r2 = R0*R0 + R1*R1 + R2*R2;

        if (r2 < rc2) {
            const double r = sqrt(r2);
            // \\exp{-B*r}
            const double exp_mbr = exp(_MB*r);

            // r^{-2, -4, -6}
            const double r_m1 = 1.0/r;
            const double r_m2 = r_m1*r_m1;
            const double r_m4 = r_m2*r_m2;
            const double r_m6 = r_m4*r_m2;

            // \\frac{C}{r^6}
            const double crm6 = _C*r_m6;

            // A \\exp{-Br} - \\frac{C}{r^6}
            u(0)+= _A*exp_mbr - crm6 + internalshift;

            // AB \\exp{-Br} - \\frac{C}{r^6}*\\frac{6}{r}
            const double term2 = crm6*(-6.0)*r_m1;
            const double f_tmp = _AB * exp_mbr + term2;

            A(0,0)+=f_tmp*R0;
            A(0,1)+=f_tmp*R1;
            A(0,2)+=f_tmp*R2;

            A(1,0)-=f_tmp*R0;
            A(1,1)-=f_tmp*R1;
            A(1,2)-=f_tmp*R2;
        }
        '''
        constants = (
                     kernel.Constant('_A', self.a),
                     kernel.Constant('_AB', self.ab),
                     kernel.Constant('_B', self.b),
                     kernel.Constant('_MB', self.mb),
                     kernel.Constant('_C', self.c),
                     kernel.Constant('rc2', self.rc ** 2),
                     kernel.Constant('internalshift', self._shift_internal)
                     )

        return kernel.Kernel('BuckinghamV',
                             kernel_code,
                             constants,
                             [kernel.Header('stdio.h'),
                              kernel.Header('math.h')])




