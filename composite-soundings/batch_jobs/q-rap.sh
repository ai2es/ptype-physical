#!/bin/bash -l
#PBS -N ptype_soundings_test
#PBS -A NAML0001
#PBS -l walltime=01:00:00
#PBS -o ptype_test.out
#PBS -e ptype_test.out
#PBS -q casper
#PBS -l select=1:ncpus=24:mem=150GB
#PBS -m a
#PBS -M dkimpara@ucar.edu
export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR
source /etc/profile.d/modules.sh
module load conda

conda activate ptype
cd /glade/u/home/dkimpara/ptype-physical/composite-soundings/batch-jobs
python -u mr-run-rap.py -i 0 -m rap -o rap