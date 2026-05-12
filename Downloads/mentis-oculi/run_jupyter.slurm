#!/bin/bash
#SBATCH --job-name=jupyter-mentis
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:ampere_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=jupyter_mentis_%j.log
#SBATCH --error=jupyter_mentis_%j.log

# Initialize conda and activate environment
eval "$(conda shell.bash hook)"
conda activate mentis-oculi

# Get the hostname of the compute node
NODE_IP=$(hostname -i)
NODE_HOSTNAME=$(hostname -s)

# Select a random port
PORT=$(shuf -i 8000-9999 -n 1)

echo -e "
==================================================================
Starting Jupyter Lab Server in conda env: mentis-oculi
==================================================================
Compute Node     : $NODE_HOSTNAME ($NODE_IP)
Port             : $PORT

To connect to this Jupyter instance, run the following SSH port
forwarding command on your LOCAL machine (laptop/desktop):

ssh -N -L ${PORT}:${NODE_HOSTNAME}:${PORT} valar

Then, open your browser and navigate to the URL printed below
(using localhost:${PORT}).
==================================================================
"

# Run Jupyter Lab from repo root
cd /scratch/eozlu21/mentis-oculi
jupyter lab --no-browser --port=${PORT} --ip=0.0.0.0
