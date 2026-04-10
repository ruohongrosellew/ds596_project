#!/bin/bash -l
#$ -P ds596
#$ -N train_squidiff_diff_data
#$ -pe omp 1
#$ -l gpus=1
#$ -M rkdesai7@bu.edu
#$ -m e

module load miniconda
module load cuda/12.8
conda activate /projectnb/ds596/projects/Team_8/squid_diff_env
cd "/projectnb/ds596/projects/Team 8/Squidiff"

python train_squidiff.py \
--logger_path /projectnb/ds596/projects/Team\ 8/process_new_data \
--data_path /projectnb/ds596/projects/Team\ 8/process_new_data/squidiff_train.h5ad \
--resume_checkpoint /projectnb/ds596/projects/Team\ 8/process_new_data \
--gene_size 200 \
--output_dim 200