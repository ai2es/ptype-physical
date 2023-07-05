#!/bin/bash -l
#PBS -N ptype_hrrr_ny2_04
#PBS -A NAML0001

### resource recs:
### rap: 1h ncpus=24:mem=250GB
### gfs: 1h ncpus=6:mem=50GB
### hrrr: 30min ncpus=32 mem=550GB
### for hrrr - only run on one case study *day* at a time (or less)

#PBS -l walltime=00:45:00
#PBS -o outfiles/ptype_hrrr_ny2_04.out
#PBS -e outfiles/ptype_hrrr_ny2_04.out
#PBS -q casper
#PBS -l select=1:ncpus=32:mem=600GB
#PBS -m a
#PBS -M dkimpara@ucar.edu
export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR
source /etc/profile.d/modules.sh
module load conda

conda activate ptype
cd /glade/u/home/dkimpara/ptype-physical/composite-soundings/batch_jobs
mkdir -p outfiles

### use following for hrrr:
python -u mr-run.py -m hrrr -o hrrr_ny2_04 -d /glade/campaign/cisl/aiml/ptype/ptype_case_studies/new_york_2/hrrr/20220204

### use folloiwng for rap and gfs
### -i 0 for kentucky, 1 for new_york_1, 2 for new_york_2
### python -u mr-run.py -i 1 -m gfs -o gfs

