from __future__ import division, print_function#, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import cgen
import ctypes
import numpy as np
import time

import build

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



class Cpp11MT19937(Module):

    def __init__(self, name=None, seed=None):
        assert name is not None, "name required"
        self.name = str(name)
        if seed is None:
            seed = int(time.time())

        header = '''
        #include <random>
        #include <memory.h>
        #include <cstdint>
        #include <iostream>
        using namespace std;
        extern "C" int get_size();
        extern "C" int get_mt_instance(uint seed, int size, mt19937 *mt_buffer);
        extern "C" uint64_t get_rand(mt19937 *mt_buffer);
        '''

        src = '''
        int get_size(){
            int size = -1;
            size = sizeof(mt19937);
            return size;
        }
        int get_mt_instance(uint seed, int size, mt19937 *mt_buffer){
            mt19937 mt_tmp(seed);
            if (sizeof(mt_tmp) != size){ return -1; }
            memcpy(mt_buffer, &mt_tmp, size);
            return 0;
        }
        // development function
        uint64_t get_rand(mt19937 *mt_buffer){
            #define foo() (mt_buffer[0]())

            const uint64_t rr = foo();
            cout << rr << endl;
            return rr;
        }
        '''

        lib = build.simple_lib_creator(
            header,
            src,
            'Cpp11MT19937Lib'
        )

        mt_size = lib['get_size']()

        assert mt_size > 0, "MT state size cannot be negative"
        self._mt_buffer = np.zeros(mt_size, dtype=ctypes.c_int8)

        mt_flag = lib['get_mt_instance'](
            ctypes.c_uint(seed),
            ctypes.c_int(mt_size),
            self._mt_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
        )
        assert mt_flag > -1, "failed to make MT instance"
        
        #for ix in range(10):
        #    val = lib['get_rand'](
        #        self._mt_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
        #    )

    def get_cpp_headers_ast(self):
        """
        Return the code to include the required header file(s).
        """
        return cgen.Module([cgen.Include('random'), cgen.Include('cstdint')])


    def get_cpp_arguments_ast(self):
        """
        Return the code to define arguments to add to the library.
        """
        return cgen.Pointer(cgen.Value('mt19937', '_mt_instance'))

    def get_cpp_pre_loop_code_ast(self):
        """
        Return the code to place before the loop.
        """
        _s = '//#undef ' + self.name + '()\n' + '#define ' + self.name + '() (_mt_instance[0]())\n'
        return cgen.Module([cgen.Line(_s)])


    def get_cpp_post_loop_code_ast(self):
        """
        Return the code to place after the loop.
        """
        _s = ''
        return cgen.Module([cgen.Line(_s)])


    def get_python_parameters(self):
        """
        Return the parameters to add to the launch of the shared library.
        """
        return self._mt_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))




