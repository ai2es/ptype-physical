#!/bin/bash -l 

#PBS -N precRAP
#PBS -l walltime=18:00:00
#PBS -l select=1:ncpus=8:ngpus=0:mem=128GB
#PBS -A NAML0001
#PBS -q casper

#PBS -j eo
#PBS -k eod

source ~/.bashrc
conda activate risk

python -u precip_RAP.py mPING True 1408 1581