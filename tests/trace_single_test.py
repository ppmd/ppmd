



import pytest
import sys

print "testing file", sys.argv[1]

pytest.main(['-svx', sys.argv[1]])




