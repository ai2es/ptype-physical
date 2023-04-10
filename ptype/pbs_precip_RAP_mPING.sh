#!/bin/bash -l 

#PBS -N precRAP
#PBS -l walltime=9:59:00
#PBS -l select=1:ncpus=8:ngpus=0:mem=256GB
#PBS -A NAML0001
#PBS -q regular

#PBS -j eo
#PBS -k eod

source ~/.bashrc
conda activate risk

python -u precip_RAP.py mPING 0 541
