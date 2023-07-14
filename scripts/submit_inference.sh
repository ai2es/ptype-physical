#!/bin/bash -l
#PBS -N ptype_inf
#PBS -A NAML0001
#PBS -l walltime=12:00:00
#PBS -o ptype_inference.out
#PBS -e ptype_inference.out
#PBS -q casper
#PBS -l select=1:ncpus=18:mem=64GB
#PBS -m a
#PBS -M cbecker@ucar.edu
conda activate ptype
cd /glade/work/cbecker/ptype/
python -u ./scripts/run_inference.py  -c ./config/inference_vaisala_NE.yml
