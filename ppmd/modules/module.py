from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

class Module(object):

    def get_cpp_headers_ast(self):
        """
        Return the code to include the required header file(s).
        """
        pass

    def get_cpp_arguments_ast(self):
        """
        Return the code to define arguments to add to the library.
        """
        pass

    def get_cpp_pre_loop_code_ast(self):
        """
        Return the code to place before the loop.
        """
        pass

    def get_cpp_post_loop_code_ast(self):
        """
        Return the code to place after the loop.
        """
        pass

    def get_python_parameters(self):
        """
        Return the parameters to add to the launch of the shared library.
        """
        pass


