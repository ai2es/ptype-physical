#!/bin/bash -l
#PBS -N ptype_hrrr
#PBS -A NAML0001

### would use this script for hrrr data
### rec resources: ncpus=32+ mem=650GB+ (not totally sure)
### only run on one case study day at a time (or less)


#PBS -l walltime=02:00:00
#PBS -o ptype_hrrr.out
#PBS -e ptype_hrrr.out
#PBS -q casper
#PBS -l select=1:ncpus=32:mem=650GB
#PBS -m a
#PBS -M dkimpara@ucar.edu
export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR
source /etc/profile.d/modules.sh
module load conda

conda activate ptype
cd /glade/u/home/dkimpara/ptype-physical/composite-soundings/batch-jobs
python -u mr-run-by-dir.py -m hrrr -o hrrr_k_0 -d /glade/campaign/cisl/aiml/ptype/ptype_case_studies/kentucky/hrrr/20220223