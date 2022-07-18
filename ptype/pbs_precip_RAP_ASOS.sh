#!/bin/bash -l 

#PBS -N precRAP
#PBS -l walltime=12:59:00
#PBS -l select=1:ncpus=8:ngpus=0:mem=256GB
#PBS -A NAML0001
#PBS -q casper

#PBS -j eo
#PBS -k eod

#module load ncarenv/1.3 gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1
source ~/.bashrc
conda activate holo_torch

python -u precip_RAP_ASOS.py 1530 1701
