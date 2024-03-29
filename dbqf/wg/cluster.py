from __future__ import print_function
import os


def ssh(sargs):
    """
    rm jobs/*.sh jobs/log/* -f && python grid_run.py --grid G --run_name X
    pattern=""; for i in 1 2; do ./kill.sh $i $pattern; done
    ./start.sh
    """
    jobs_0 = ['machine0_gpu0', 'machine0_gpu1',
              'machine1_gpu0', 'machine1_gpu1',
              ]
    # validate start.sh
    njobs = [2]*4  # Number of parallel jobs on each machine
    jobs = []
    for s, n in zip(jobs_0, njobs):
        jobs += ['%s_job%d' % (s, i) for i in range(n)]
    parallel = False  # each script runs in sequence
    return jobs, parallel


def slurm(sargs, prefix, njobs):
    """
    rm jobs/*.sh jobs/log/* -f && python grid_run.py --grid G --run_name X \
    --cluster_args <njobs>,<ntasks>,<partitions>
    pattern=""; for i in 1 2; do ./kill.sh $i $pattern; done
    sbatch jobs/slurm.sbatch
    """
    ngpu, nnodes, partition = sargs.split(',', 2)
    # njobs = 5  # Number of array jobs
    # ntasks = 4  # Number of running jobs
    jobs = [str(i) for i in range(njobs)]
    sbatch_f = """#!/bin/bash

#SBATCH --job-name=array
#SBATCH --output=jobs/log/array_%A_%a.log
#SBATCH --array=0-{njobs}
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH -c 3
#SBATCH --mem=18G
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=iman.tabrizian+slurm@gmail.com
#SBATCH -p {partition}

date; hostname; pwd
python -c "import torch; print(torch.__version__)"
(while true; do nvidia-smi; top -b -n 1 | head -20; sleep 10; done) &

# the environment variable SLURM_ARRAY_TASK_ID contains
# the index corresponding to the current job step
source $HOME/iman/dist-quantization/dbqf.sh
srun --mem 18G bash jobs/{prefix}/$SLURM_ARRAY_TASK_ID.sh
""".format(njobs=njobs-1, partition=partition, prefix=prefix,
           )
    try:
        os.mkdir(f'jobs/{prefix}')
    except OSError:
        pass
    with open('jobs/' + prefix + '/slurm.sbatch', 'w') as f:
        print(sbatch_f, file=f)
    return jobs
