rsync -avz ~/Developer/hpc-project/ uni-server:~/hpc-project/

cd ~/hpc-project
module load amd/gcc/gcc-12
module load amd/gcc-12.2.1/openmpi-4.1.6
module load amd/nvidia/cuda-12.3.2
make all NVCC_CCBIN=g++
