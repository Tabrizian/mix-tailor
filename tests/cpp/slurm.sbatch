#!/bin/bash

#SBATCH -p p100
#SBATCH --nodes=4               # request to distribute the tasks across four nodes
#SBATCH --ntasks=4              # number of tasks (mpi processes) to run 
#SBATCH --mem=8G                # amount of memory to allocate per node
#SBATCH --gpus-per-task=1       # number of GPUs we want per task
#SBATCH --cpus-per-gpu=4        # number of CPUs we want per GPU
#SBATCH --time=3:00             # run no longer than 3h


# load the mpi module (setup environment)
module load mpich-3.3.2

srun --mem 8G --mpi=pmi2 b &

wait