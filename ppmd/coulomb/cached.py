from __future__ import division, print_function, absolute_import
__author__ = "W.R.Saunders"
__copyright__ = "Copyright 2016, W.R.Saunders"
__license__ = "GPL"

from functools import wraps
class _old_cached(object):
    def __init__(self, maxsize=None):
        pass
    def __call__(self, function):
        # pythontips.com
        memo = {}
        @wraps(function)
        def wrapper(*args, **kwargs):
            if len(kwargs) > 0:
                return function(*args, **kwargs)

            if args in memo:
                return memo[args]
            else:
                rv = function(*args)
                memo[args] = rv
                return rv


        return wrapper
try:
    from functools import lru_cache
    cached = lru_cache
except Exception as e:
    cached = _old_cached















