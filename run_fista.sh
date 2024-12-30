#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=1:00:00
#$ -N run_fista
#$ -o output.log
#$ -e error.log

python estimate_h.py
