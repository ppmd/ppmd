import re


class Constant(object):
    """Class representing a numerical constant.

    This class can be used to use placeholders for constants
    in kernels.
    
    :arg str name: Name of constant
    :arg value: Numerical value (can actually be any data type)
    """

    def __init__(self, name, value):
        self._name = name
        self._value = value

    def replace(self, s):
        """Replace all occurances in a string and return result
        
        Ignores the constant if it is not a C-variable. For example,
        if the name of the constant is ``mass``, then it would not replace
        it in ``mass1`` or ``Hmass``.

        :arg str s: string to work on
        """

        # forbiddenChars='[^a-zA-Z0-9_]' #='[\W]'='[^\w]'
        
        forbiddenchars = '[\W]'
        regex = '(?<='+forbiddenchars+')('+self._name+')(?='+forbiddenchars+')'
        
        return re.sub(regex, str(self._value), s)
