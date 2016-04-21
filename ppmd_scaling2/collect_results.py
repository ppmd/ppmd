#!/usr/bin/python

import os

# everyone likes global vars
cwd = os.getcwd()
config_dir = os.path.join(cwd, 'config_dir')
jobscript_dir = os.path.join(cwd, 'jobscript_dir')

record_ppmd = os.path.join(cwd, 'record_ppmd')
record_lmps = os.path.join(cwd, 'record_lmps')
record_dlpoly = os.path.join(cwd, 'record_dlpoly')

_record_ppmd = []
_record_lmps = []
_record_dlpoly = []


def time_parse_ppmd(file_in):
    fh = open(file_in, 'r')
    output = fh.read()
    fh.close()
    output = [l for l in output.splitlines() if l.startswith("Velocity Verlet:")]
    
    if len(output) != 1:
        print "PPMD WARNING no time found in", file_in, "returning"
        return -1.0
    
    # get the time from the string
    output = float(output[0].split(" ")[-2])
    return output

def time_parse_lmps(file_in):
    fh = open(file_in, 'r')
    output = fh.read()
    fh.close()

    output = [l for l in output.splitlines() if l.startswith("Loop time of ")]
    
    if len(output) != 1:
        print "LAMMPS WARNING no time found in", file_in, "returning"
        return -1.0
    
    # get the time from the string
    output = float(output[0].split(" ")[3])
    return output

    return -1

def time_parse_dlpoly(file_in):
    fh = open(file_in, 'r')
    output = fh.read()
    fh.close()
    
    output = output.splitlines()
    
    time_lines = []


    for ln, ll in enumerate(output):
        if ll.startswith("      time(ps)      eng_pv    temp_rot"):
            time_lines.append(ln + 6)
    
    if len(time_lines) != 3:
        print "DLPOLY WARNING unexpected number of time lines:", len(time_lines)
        return -1

    line1 = output[time_lines[1]].split()[0]
    line2 = output[time_lines[2]].split()[0]
    
    return float(line2) - float(line1)




def collect_all():
    source_dirs = [f for f in os.listdir(cwd) if (os.path.isdir(os.path.join(cwd, f)) and f.startswith("run_") )]

    for _dir in source_dirs:
        print "Processing", _dir
        
        _split = _dir.split("_")
        _nn = int(_split[1])
        _nc = int(_split[2])
        
        print "\t", _nn, "nodes,", _nc, "cores per node =", _nn*_nc, "cores"

        abs_dir = os.path.join(cwd, _dir)

        # read times for ppmd
        ppmd_files = [f for f in os.listdir(abs_dir) if (os.path.isfile(os.path.join(abs_dir, f)) and f.startswith("pfprint_") )]
        print "\t", "found ppmd files:", ppmd_files

        for _file in ppmd_files:
            _time = time_parse_ppmd(os.path.join(abs_dir, _file))

            # check time is not an error
            if _time >= -0.5:
                print "\t\trecording ppmd time:", _time
                _record_ppmd.append((_nn*_nc, _time))
        
        # read times for lammps
        _time = time_parse_lmps(os.path.join(abs_dir, 'log.lammps'))

        # check time is not an error
        if _time >= -0.5:
            print "\t\trecording lammps time:", _time
            _record_lmps.append((_nn*_nc, _time))

        # read times for dlpoly
        _time = time_parse_dlpoly(os.path.join(abs_dir, 'OUTPUT'))

        # check time is not an error
        if _time >= -0.5:
            print "\t\trecording dlpoly time:", _time
            _record_dlpoly.append((_nn*_nc, _time))


if __name__ == "__main__":
    
    # collect records
    collect_all()

    # sort by core count
    _record_ppmd.sort(key=lambda x:x[0])
    _record_lmps.sort(key=lambda x:x[0])
    _record_dlpoly.sort(key=lambda x:x[0])

    #write records
    fh = open(record_ppmd, 'w')
    for r in _record_ppmd:
        fh.write("%(NC)s %(TIME)s\n" % {'NC': str(r[0]), 'TIME': str(r[1])})
    fh.close()

    fh = open(record_lmps, 'w')
    for r in _record_lmps:
        fh.write("%(NC)s %(TIME)s\n" % {'NC': str(r[0]), 'TIME': str(r[1])})
    fh.close()

    fh = open(record_dlpoly, 'w')
    for r in _record_dlpoly:
        fh.write("%(NC)s %(TIME)s\n" % {'NC': str(r[0]), 'TIME': str(r[1])})
    fh.close()

















