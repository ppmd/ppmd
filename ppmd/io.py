# System level
import os
import xml.etree.cElementTree as ET
from xml.dom import minidom
import ctypes
import numpy as np

# This has fancy printing but has issues importing on mapc-4044
#from lxml import etree as ET
# import lxml.etree as ET

import datetime


# package level
import data

##########################################################################
# XML Fun
##########################################################################


TYPE_MAP = {'c_double': ctypes.c_double,
            'c_int': ctypes.c_int}


def human_readable(E):
    """Return a pretty-printed XML string for the Element.
    """
    raw = ET.tostring(E, 'utf-8')
    formatted = minidom.parseString(raw)
    return formatted.toprettyxml(indent="\t")




def ParticleDat_to_xml(dat=None, filename=None):
    """
    Write a particle dat to disk in XML format
    :param dat:
    :return:
    """


    assert type(dat) is data.ParticleDat, "No/incorrect ParticleDat type"
    assert type(filename) is str, "No/incorrect filename type"

    root = ET.Element('ParticleDat')
    root.append(ET.Comment('ParticleDat record'))

    timestamp = ET.SubElement(root, 'timestamp')
    timestamp.text = 'timestamp: {0}'.format(datetime.datetime.now().isoformat())

    dtype = ET.SubElement(root,'dtype')
    dtype.text = str(dat.dtype.__name__)

    npart = ET.SubElement(root,'npart')
    npart.text = str(dat.npart)

    ncomp = ET.SubElement(root,'ncomp')
    ncomp.text = str(dat.ncomp)

    name = ET.SubElement(root,'name')
    name.text = str(dat.name)


    fh = open(filename,'w')
    xml_str = human_readable(root)
    fh.writelines('XML_START 1 XML_END ' + str(len(xml_str.splitlines())) + '\n')
    fh.write(xml_str)
    fh.close()

    fh = open(filename,'a')
    dat.dat.tofile(fh)
    fh.close()




def xml_to_ParticleDat(filename=None):
    """
    Read a file into a particle dat
    :param filename:
    :return:
    """
    assert type(filename) is str, "No filename passed."
    assert os.path.isfile(filename), "File not found"

    fh = open(filename, 'r')

    xml_start_end = fh.readline().split()
    xml_end = int(xml_start_end[xml_start_end.index('XML_END') + 1])


    xml =''
    for fx in range(xml_end):
        xml += fh.readline()

    root = ET.fromstring(xml)

    # Check this file is a ParticleDat container
    assert root.tag == 'ParticleDat', "This is not a ParticleDat file."

    # Read attirbutes
    children = {}
    for child in root:
       children[child.tag] = child.text

    dat = data.ParticleDat(ncomp=int(children['ncomp']),
                           npart=int(children['npart']),
                           name=children['name'],
                           dtype=TYPE_MAP[children['dtype']])

    dat.dat = np.reshape(np.fromfile(fh, dtype=TYPE_MAP[children['dtype']], count=dat.ncomp * dat.npart),
                         [dat.npart,dat.ncomp])

    return dat






