#!/bin/bash -l
#PBS -N ptype_aws
#PBS -A NAML0001
#PBS -l walltime=17:00:00
#PBS -o aws_up.out
#PBS -e aws_up.out
#PBS -q casper
#PBS -l select=1:ncpus=1:mem=16GB
#PBS -m a
#PBS -M dgagne@ucar.edu
cd /glade/scratch/dgagne/
aws s3 cp --recursive --acl public-read vaisala_NE s3://ncar-ml-ptype/vaisala_NE
