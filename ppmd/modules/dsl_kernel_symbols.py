"""
regex alter kernel symbols
"""

from __future__ import division, print_function, absolute_import

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

import ctypes
from cgen import *
import re


class DSLKernelSymSub(object):
    def __init__(self, kernel):
        self.kernel = kernel

    def sub_sym(self, old_sym, new_sym):

        fchars = "[^a-zA-Z0-9_]"  # ='[\W]'='[^\w]'
        regex = "(?<=" + fchars + ")(" + re.escape(old_sym) + ")"

        k = re.sub(regex, new_sym, self.kernel)
        self.kernel = k

    @property
    def code(self):
        return self.kernel
