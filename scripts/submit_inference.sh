#!/bin/bash -l
#PBS -N ptype_inf
#PBS -A NAML0001
#PBS -l walltime=00:30:00
#PBS -o ptype_inference.out
#PBS -e ptype_inference.out
#PBS -q main
#PBS -l select=1:ncpus=20:ngpus=4:mem=128GB -l gpu_type=a100
#PBS -m a
#PBS -M cbceker@ucar.edu
conda activate ptype
cd /glade/work/cbecker/ptype-physical/
python -u ./scripts/run_inference.py  -c ./config/inference.yml
