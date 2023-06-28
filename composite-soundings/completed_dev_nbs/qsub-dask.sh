#!/bin/bash -l
#PBS -N ptype_composite_soundings
#PBS -A NAML0001
#PBS -l walltime=08:00:00
#PBS -o ptype_soundings.out
#PBS -e ptype_soundings.out
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=32GB
#PBS -m a
#PBS -M dkimpara@ucar.edu
export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR
source /etc/profile.d/modules.sh
module load conda

conda activate ptype
cd /glade/u/home/dkimpara/ptype-physical/composite-soundings/batch-jobs
python -u dask-qsub.py