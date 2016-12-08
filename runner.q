#!/bin/bash 

#PBS -V
#PBS -N test
#PBS -l nodes=1:ppn=1:gpus=1:titan,mem=20GB,walltime=7:00:00
#PBS -j oe

# echo $CUDA_VISIBLE_DEVICES
cd /home/pks329/traffic-sign-detection/
module purge
module load torch/gnu/20160623
# module list
 
qlua jitter.lua -model vgg -nEpochs 100 >> /home/pks329/traffic-sign-detection/vgg100_jitter_lr1.log
