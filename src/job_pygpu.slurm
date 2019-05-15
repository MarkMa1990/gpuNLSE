#!/bin/bash

#SBATCH -p GPU
#SBATCH -J test
#SBATCH --nodes=1
#SBATCH --ntasks=1


###SBATCH --gres=gpu:titanxm:1
#SBATCH --gres=gpu:1

#SBATCH --exclude=calcul-gpu-lahc-2


#SBATCH --mem=2GB
#SBATCH --time=1:30:00
#SBATCH --output outtest.out

source /home_expes/tools/python/python367_gpu

##ls /home_expes/tools/cuda/cuda-10.0

echo "on node: " $SLURMD_NODENAME
srun --exclusive nvidia-smi
echo ""

cmake --version
cmake -DBOOST_DIR=/home_expes/tools/boost/boost_1_66_0 .

make

#   srun --exclusive ./routines 2 
#   srun --exclusive ./routines 4 
#   srun --exclusive ./routines 6 
#   srun --exclusive ./routines 8 
#   
#   srun --exclusive ./routines 16 
#   srun --exclusive ./routines 24 
#   srun --exclusive ./routines 32 
#   
#   srun --exclusive ./routines 64 
#   srun --exclusive ./routines 128 
#   srun --exclusive ./routines 256 

#   srun --exclusive ./routines 32
srun --exclusive ./main_loop
