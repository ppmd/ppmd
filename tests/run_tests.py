

from subprocess import Popen, STDOUT, PIPE
from glob import glob
import sys
import os




def run_parallel(filename, nproc):

    cmd = (
        'mpirun',
        '-n',
        str(nproc),
        'py.test',
        '-svx',
        filename
    )
    
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    
    if p.returncode:
        print(filename, "RETURNED NON-ZERO ERRORCODE")
        print(out.decode(sys.stdout.encoding))
        print(err.decode(sys.stdout.encoding))
        return(filename, nproc)
    else:
        print(filename, "PASSED, N_PROC:", nproc)
        return False



if __name__ == '__main__':
    

    nproc_list = [1]

    for nproc in (2,4,8):
        p = Popen(('mpirun', '-n', str(nproc), 'hostname'), stdout=PIPE, stderr=PIPE)
        _, _ = p.communicate()
        if not p.returncode:
            nproc_list.append(nproc)
    

    dirs = sys.argv[1:]

    if len(dirs) == 0:
        dirs = (
            'host',
            'host/fmm',
            'cuda'
        )
        
    
    failures = []

    for dx in dirs:

        filenames = glob(os.path.join(dx, 'test*.py'))

        for fx in filenames:
            
            for nproc in nproc_list:
                r = run_parallel(fx, nproc)
                if r:
                    failures.append(r)


    if len(failures) > 0:
        print("-" * 80)
        print("Failures: (filename, nproc)")
        print("-" * 80)
        for fx in failures:
            print(fx)
        sys.exit(1)






















