#!/bin/bash

#SBATCH -p LONG
#SBATCH -J test_getdata
#SBATCH --nodes=1
#SBATCH --ntasks=1


###SBATCH --gres=gpu:titanxm:1
##SBATCH --gres=gpu:0

##SBATCH --exclude=calcul-gpu-lahc-2


#SBATCH --mem=2GB
#SBATCH --time=1:30:00
#SBATCH --output outtest_getdata.out

#source /home_expes/tools/python/python367_gpu

##ls /home_expes/tools/cuda/cuda-10.0

echo "on node: " $SLURMD_NODENAME
echo ""

srun --exclude python3 getdata.py

