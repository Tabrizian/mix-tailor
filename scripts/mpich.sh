#!/bin/bash

#SBATCH --nodes=5
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=5G
#SBATCH --output=jobs/log/%J.out
#SBATCH --time=0-02:00:00

srun --mem 5G scripts/bash.sh
