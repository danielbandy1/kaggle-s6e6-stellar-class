#!/bin/bash
#SBATCH --account=coa_ich248_uksr
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:30:00
#SBATCH --job-name=s6e6_tune
#SBATCH --output=/home/dkba237/kaggle/s6e6/slurm_%j.out

set -euo pipefail

cd /home/dkba237/kaggle/s6e6

if [ -d .venv ]; then
  source .venv/bin/activate
fi

python3 -m pip install --quiet --user pandas numpy scikit-learn lightgbm optuna
python3 optuna_lgb_tune.py
