#!/bin/bash -l
#PBS -N ptype_gfs
#PBS -A NAML0001

### resource recs:
### rap: 1h ncpus=23:mem=250GB
### gfs: 1h ncpus=6:mem=50GB
### would not run hrrr with this script

#PBS -l walltime=01:00:00
#PBS -o ptype_gfs.out
#PBS -e ptype_gfs.out
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=50GB
#PBS -m a
#PBS -M dkimpara@ucar.edu
export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR
source /etc/profile.d/modules.sh
module load conda

conda activate ptype
cd /glade/u/home/dkimpara/ptype-physical/composite-soundings/batch-jobs

### Run analysis script -i 0 for kentucky, 1 for new_york_1, 2 for new_york_2
python -u mr-run-by-case.py -i 1 -m gfs -o gfs