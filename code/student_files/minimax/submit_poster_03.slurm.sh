#!/usr/bin/env bash
# Separate MiniMax-H3 experiment on poster_03. [AI-CODE]
#SBATCH --job-name=h3-poster03
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=h3-poster03-%j.log

set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"
LAB_ROOT="${LAB_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
: "${MINIMAX_ENV:?Export MINIMAX_ENV to the MiniMax Conda environment prefix}"
: "${MINIMAX_WEIGHTS:?Export MINIMAX_WEIGHTS to the downloaded Ref2VA checkpoint}"

export PATH="$MINIMAX_ENV/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

cd "$LAB_ROOT"
command -v ffmpeg
command -v ffprobe
test -f "$MINIMAX_WEIGHTS/lab_snapshot.json"
printf 'Job %s on %s\n' "$SLURM_JOB_ID" "$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

exec "$MINIMAX_ENV/bin/python" code/student_files/minimax/run_ref2va.py run \
  --weights "$MINIMAX_WEIGHTS" \
  --output-dir "$LAB_ROOT/code/outputs/genai/minimax_h3_poster_03"
