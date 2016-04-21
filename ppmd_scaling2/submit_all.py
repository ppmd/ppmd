#!/usr/bin/python

import os


cwd = os.getcwd()
jobscript_dir = os.path.join(cwd, 'jobscript_dir')


def submit_all():
    print "submitting jobs from", jobscript_dir
    os.chdir(jobscript_dir)
    files = [f for f in os.listdir(jobscript_dir) if os.path.isfile(os.path.join(jobscript_dir, f))]
    
    for f in files:
        print "running sbatch", f
        os.system('sbatch ' + str(f))


    os.chdir(cwd)






if __name__ == "__main__":
    submit_all()



