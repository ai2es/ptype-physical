#!/bin/bash -l
#PBS -N ptype_hrrr
#PBS -A NAML0001

### resource recs:
### rap: 1h ncpus=23:mem=250GB
### gfs: 1h ncpus=6:mem=50GB
### hrrr: ncpus=32+ mem=650GB+ (not totally sure)
### for hrrr - only run on one case study *day* at a time (or less)

#PBS -l walltime=02:00:00
#PBS -o ptype_hrrrk24.out
#PBS -e ptype_hrrrk24.out
#PBS -q casper
#PBS -l select=1:ncpus=32:mem=650GB
#PBS -m a
#PBS -M dkimpara@ucar.edu
export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR
source /etc/profile.d/modules.sh
module load conda

conda activate ptype
cd /glade/u/home/dkimpara/ptype-physical/
git checkout dkimpara
cd composite-soundings/batch_jobs

### use following for hrrr:
python -u mr-run.py -m hrrr -o hrrr_k_24 -d /glade/campaign/cisl/aiml/ptype/ptype_case_studies/kentucky/hrrr/20220224

### use folloiwng for rap and gfs
### -i 0 for kentucky, 1 for new_york_1, 2 for new_york_2
### python -u mr-run.py -i 1 -m gfs -o gfs

