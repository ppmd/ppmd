#!/usr/bin/python
"""
This script creates rst sources files from rst embeded within the python source file.

The desired contents of the modules rst file must be enclosed as "rst_doc{rst text here}rst_doc. Only the first occurance will be taken.
"""

import getopt
import sys
import os
import re

input_dir = None
output_dir = None


try:
    opts, args = getopt.getopt(sys.argv[1:], "I:O:")
except getopt.GetoptError as err:
    # print help information and exit:
    print(str(err)) # will print something like "option -a not recognized"
    sys.exit(2)
for o, a in opts:
    if o == "-I":
        input_dir=os.path.abspath(a)
    elif o == "-O":
        output_dir=os.path.abspath(a)
    else:
        assert False, "unhandled option"


assert input_dir is not None, "Must specify an input directory."
assert output_dir is not None, "Must specify an output directory."


for file_id in os.listdir(input_dir):
    if file_id.endswith(".py"):
        fh = open(os.path.join(input_dir, file_id),'r')
        file_contents = fh.read()
        fh.close()
        
        doc_str = ''.join(re.findall('(?<=rst_doc\{)(.*)(?=\}rst_doc)', file_contents, re.DOTALL))
        
        if doc_str is not '':
            
            fh = open(os.path.join(output_dir, file_id[:-2] + 'rst'),'w')
            fh.write(doc_str)
            fh.close()
            




























