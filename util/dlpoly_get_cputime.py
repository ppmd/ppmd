#!/usr/bin/python
import re





fh=open('OUTPUT')
line_count = -1
int_time = -1
for i, line in enumerate(fh):
    items=re.findall("cpu",line,re.MULTILINE)
    if (len(items)>0):
        line_count = i+5
    elif (i==line_count):
        if(len(line.strip().split())>0):
            int_time = line.strip().split()[0]
fh.close()
print "DL_POLY time taken ",int_time, "s"









