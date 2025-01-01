#!/bin/bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=1:00:00
#$ -N run_fista
#$ -o output.log
#$ -e error.log

module load cuda/12.3.2

python estimate_h_split.py
