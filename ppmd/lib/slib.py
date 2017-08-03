from __future__ import print_function, division

import ppmd.opt
import ppmd.runtime

__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"

from ppmd import opt

class Lib(object):
    def __init__(self, name,lib):
        self.name = name
        self.lib = lib
        self.timer = ppmd.opt.Timer()

    def __call__(self, *args):
        self.timer.start()
        ret = self.execute_no_time(*args)
        self.timer.pause()
        opt.PROFILE[self.name] = self.timer.time()
        return ret

    def execute_no_time(self, *args):
        return self.lib(*args)
