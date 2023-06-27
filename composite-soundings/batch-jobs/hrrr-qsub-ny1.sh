#!/bin/bash -l
#PBS -N ptype_composite_soundings
#PBS -A NAML0001
#PBS -l walltime=06:00:00
#PBS -o ptype-mr.out
#PBS -e ptype-mr.out
#PBS -q casper
#PBS -l select=1:ncpus=16:mem=70GB
#PBS -m a
#PBS -M dkimpara@ucar.edu
export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR
source /etc/profile.d/modules.sh
module load conda

conda activate ptype
cd /glade/u/home/dkimpara/ptype-physical/composite-soundings/batch-jobs
python -u map-red-qsub.py -m hrrr -o hrrr_ny1 -d /glade/campaign/cisl/aiml/ptype/ptype_case_studies/new_york_1
