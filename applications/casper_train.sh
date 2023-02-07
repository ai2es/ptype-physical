#!/bin/bash -l
#PBS -N mode_cnn
#PBS -A NAML0001
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o train_ptype.out
#PBS -q casper
#PBS -l select=1:ncpus=8:ngpus=1:mem=64GB -l gpu_type=v100

module load cuda/11 cudnn
conda activate ptype
python -u ./applications/train_mlp.py -c config/ptype.yml
python -u ./applications/evaluate_mlp.py -c config/ptype.yml