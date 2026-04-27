#!/bin/bash -l
#$ -P ds596
#$ -N train_squidiff_diff_data
#$ -pe omp 1
#$ -l gpus=1
#$ -M kashishg@bu.edu
#$ -m e

module load miniconda
module load cuda/12.8
conda activate /projectnb/ds596/projects/Team_8/squid_diff_env
cd "/projectnb/ds596/projects/Team 8/Squidiff2"

python train_squidiff.py \
--logger_path /projectnb/ds596/projects/Team\ 8/activation_extension \
--data_path /projectnb/ds596/projects/Team\ 8/activation_extension/squidiff_train.h5ad \
--gene_size 200 \
--output_dim 200 \
--activation relu