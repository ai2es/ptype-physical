#!/bin/bash -l
#PBS -N ptype_composite_soundings
#PBS -A NAML0001
#PBS -l walltime=05:00:00
#PBS -o ptype-mr.out
#PBS -e ptype.out
#PBS -q casper
#PBS -l select=1:ncpus=32:mem=100GB
#PBS -m a
#PBS -M dkimpara@ucar.edu
export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR
source /etc/profile.d/modules.sh
module load conda

conda activate ptype
cd /glade/u/home/dkimpara/ptype-physical/composite-soundings/batch-jobs
python -u map-red-qsub.py rap rap0
