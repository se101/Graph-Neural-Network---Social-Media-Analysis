#!/bin/bash
#SBATCH -p gpu_v100_2
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --job-name=myJobarrayTest
#SBATCH --time=13:00:00
#SBATCH --mem=4G
#SBATCH --output /scratch/dipanjan/Twitter_Analysis/slurm/slurm.%j.out
#SBATCH --error /scratch/dipanjan/Twitter_Analysis/slurm/slurm.%j.err
#SBATCH --mail-user=f20191207@hyderabad.bits-pilani.ac.in
#SBATCH --mail-type=ALL

#module purge
#module --ignore_cache load python/intel/2.7.12

#NAME=1.csv
DATADIR=/scratch/dipanjan/Twitter_Analysis/reTEST100/retweets
OUTPUTDIR=/scratch/dipanjan/Twitter_Analysis/reEmbeddings

spack load anaconda3
conda init bash
eval "$(conda shell.bash hook)"
conda activate /home/dipanjan/.conda/envs/network_venv

cd /scratch/dipanjan/Twitter_Analysis
#python sample_count.py 
#python Embeddings.py $DATADIR/$NAME $DATADIR/$NAME.$SLURM_JOB_ID.out
python reEmbeddings.py $DATADIR/$SLURM_ARRAY_TASK_ID.csv $OUTPUTDIR/$SLURM_ARRAY_TASK_ID.$SLURM_JOB_ID.csv
#sample-$SLURM_ARRAY_TASK_ID.csv
