#!/bin/bash -l
#PBS -N ptype_soundings_test
#PBS -A NAML0001
#PBS -l walltime=00:10:00
#PBS -o ptype-mr.out
#PBS -e ptype-mr.out
#PBS -q casper
#PBS -l select=1:ncpus=3:mem=12GB
#PBS -m a
#PBS -M dkimpara@ucar.edu
export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR
source /etc/profile.d/modules.sh
module load conda

conda activate ptype
cd /glade/u/home/dkimpara/ptype-physical/composite-soundings/batch-jobs
python -u mr-test.py -o step_test
