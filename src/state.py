import data


class AsFunc(object):
    """
    Instances of this class provide a callable to return the value of an attribute within an
    instance of another class.
    """

    def __init__(self, instance, name):
        self._i = instance
        self._n = name

    def __call__(self):
        return getattr(self._i, self._n)



class BaseMDState(object):
    """
    Create an empty state to which particle properties such as position, velocities are added as
    attributes.
    """

    def __init__(self):
        # Registered particle dats.
        self.particle_dats = []

        # Local number of particles
        self._n = 0

    def __setattr__(self, name, value):
        """
        Works the same as the default __setattr__ except that particle dats are registered upon being
        added. Added particle dats are registered in self.particle_dats.
        :param name: Name of parameter.
        :param value: Value of parameter.
        :return:
        """

        # Add to instance list of particle dats.
        if type(value) is data.ParticleDat:
            object.__setattr__(self, name, value)
            self.particle_dats.append(name)

        # Any other attribute.
        else:
            object.__setattr__(self, name, value)

    def as_func(self, name):
        """
        Returns a function handle to evaluate the required attribute.
        :param string name: Name of attribute.
        :return: Function handle (of type class: AsFunc)
        """
        return AsFunc(self, name)

    @property
    def n(self):
        """
        Return local number of particles.
        :return:
        """
        return self._n

    @n.setter
    def n(self, value):
        """
        Set local number of particles.
        :param value: New number of local particles.
        :return:
        """
        self._n = int(value)
        for ix in self.particle_dats:
            _dat = getattr(self,ix)
            _dat.npart = int(value)
            _dat.halo_start_reset()


