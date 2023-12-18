#!/bin/bash
#SBATCH --job-name=bert_train
#SBATCH --output=bert_train
#SBATCH --gres=gpu:1


cd /local/scratch/achun46
source venv/bin/activate
python BERT_BEST.py