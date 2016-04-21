#!/usr/bin/python

import os, getopt, sys, shutil, math, datetime

runs = (
        (1,1),
        (1,4),
        (1,8),
        (1,16),
        (2,16),
        (4,16)
        )


cwd = os.getcwd()
config_dir = os.path.join(cwd, 'config_dir')
jobscript_dir = os.path.join(cwd, 'jobscript_dir')
time_base = 2

def make_config():

    F = 'SIMULATION_RECORD'

    try:
        opts, args = getopt.getopt(sys.argv[1:], "F:")
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        sys.exit(2)
    for o, a in opts:
        if o == "-F":
            F=str(a)
        else:
            assert False, "unhandled option"

    if os.path.isfile(F):
        print "Using:", F
    else:
        print "ERROR File not found:", F
        quit()

    files = (F, 'SIMULATION_RECORD', 'CONFIG', 'CONTROL', 'FIELD', 'lammps_data.lmps', 'lammps_input.lmps')

    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    shutil.copyfile(F, os.path.join(config_dir, F))

    os.chdir(config_dir)

    os.system('python ../dlpoly_argon_lmps_generator.py -F %(NAME)s' % {'NAME':F})

    for _f in files:    
        assert os.path.isfile(_f), "ERROR one or more config files missing."

    os.chdir(cwd)

    return files


def make_run():

    print "Tuples of (nodes, cores):", runs
    
    files = make_config()

    if not os.path.exists(jobscript_dir):
        os.makedirs(jobscript_dir)

    for run in runs:

        # get folder name
        folder_name = os.path.join(cwd, 'run_' + str(run[0]) + '_' + str(run[1]))

        # make folder if it does not exist.
        if not os.path.exists(folder_name):
            print "Creating folder:" , folder_name
            os.makedirs(folder_name)

        # copy generated files
        for _f in files:    
            shutil.copyfile(os.path.join(config_dir, _f), os.path.join(folder_name, _f))
        
        time_est = int(math.ceil ( (float(time_base) * 2.0 / (run[0] * run[1])) ))

        # ppmd jobscript
        make_jobscript('ppmd_' + str(run[0]) + '_' + str(run[1]),
                       folder_name,'python ../validation_test_timing.py',run[0],run[1], time_est)
        
        # lammps jobscript
        make_jobscript('lmps_' + str(run[0]) + '_' + str(run[1]),
                       folder_name,'lmp_mpi -in lammps_input.lmps',run[0],run[1], time_est)

        # dlpoly jobscript
        make_jobscript('dlpoly_' + str(run[0]) + '_' + str(run[1]),
                       folder_name,'DLPOLY.Z',run[0],run[1], 4*time_est)

def make_jobscript(se, wd, cmd, nn, nc, et, acc="free"):
    """
    :arg se: script extension, append jobscript name
    :arg wd: working directory for job
    :arg cmd: command to execute with mpirun
    :arg nn: number of nodes
    :arg nc: number of cores per node
    :arg et: estimated time for job (minutes)
    :arg acc: account to use (default "free")
    """

    _time = datetime.timedelta(minutes=et+2)
    _time = datetime.datetime(1,1,1) + _time

    ET = "%02d:%02d:%02d" % (_time.hour, _time.minute, _time.second)

    _d = {'WD': str(wd),
          'CMD': str(cmd),
          'NN': str(nn),
          'NC': str(nc),
          'ET': str(ET),
          'ACC': str(acc)
         }


    base = '''#!/bin/bash
     
# set the account to be used for the job
#SBATCH --account=%(ACC)s

export EXEC="%(CMD)s"

#SBATCH --workdir="%(WD)s"

#export vars
#SBATCH --export=ALL

NAME="$SLURM_JOBID"

# set name of job
#SBATCH --job-name=ppmd_test
#SBATCH --output=job.%%J.out
#SBATCH --error=job.%%J.err

# set the number of nodes
#SBATCH --nodes=%(NN)s
#SBATCH --ntasks-per-node=%(NC)s
 
#use gpus
#SBATCH --partition=batch-all
##SBATCH --constraint=k20x

# set max wallclock time
####SBATCH --time=00:05:00
#SBATCH --time=%(ET)s


#print info
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"
echo "$SLURM_SUBMIT_DIR"
echo "mpirun -np $SLURM_NTASKS $EXEC"


# run the application
mpirun -n $SLURM_NTASKS $EXEC

''' % _d
    
    # end of base jobscript
    
    _fh = open(os.path.join(jobscript_dir, 'jobscript_' + str(se)),'w')
    _fh.write(base)
    _fh.close()



if __name__ == "__main__":
    make_run()
