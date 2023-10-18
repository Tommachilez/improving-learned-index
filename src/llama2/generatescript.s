#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx8000
#SBATCH --mem=64GB
#SBATCH --time=30:00:00
#SBATCH --job-name=llama2_inference
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soyuj@nyu.edu
#SBATCH --output=llama2_inference_%j.out

module load cuda/11.6.2
module load cudnn/8.6.0.163-cuda11
cd $HOME/improving-learned-index
python -m src.llama2.generate_using_merged \
  --collection_path /scratch/sjb8193/positive_collection.tsv \
  --output_path /hdd1/home/soyuj/positive_queries.tsv \
  --batch_size 1