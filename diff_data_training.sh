#!/bin/bash -l
#$ -P ds596
#$ -N train_squidiff_diff_data
#$ -pe omp 1
#$ -l gpus=1

$path_to_conda="/path_to_conda_env"
$working_dir="/path_to_working_directory"
$logger_path="/path_to_logger_directory"
$data_path="/path_to_data_file.h5ad"
$resume_checkpoint="/path_to_resume_checkpoint"

module load miniconda
module load cuda/12.8
mamba activate $path_to_conda
cd $working_dir

python train_squidiff.py \
--logger_path $logger_path \
--data_path $data_path \
--resume_checkpoint $resume_checkpoint \
--gene_size 203 \
--output_dim 203