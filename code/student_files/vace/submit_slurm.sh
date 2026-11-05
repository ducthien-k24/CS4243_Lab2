#!/usr/bin/env bash
# Example only: select the actual NUS account/partition at sbatch time. [AI-CODE]
#SBATCH --job-name=lab2-vace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=vace-%j.log

set -euo pipefail
: "${LAB_ROOT:?Set absolute repository path}"
: "${VACE_ENV:?Set absolute environment path}"
: "${VACE_WEIGHTS:?Set absolute checkpoint path}"
: "${VACE_VIDEO:?Set original full personal clip.mp4 path}"
: "${VACE_ARTWORK:?Set own artwork PNG path}"
: "${VACE_MASKS:?Set Notebook 4 exported board-mask directory}"
: "${VACE_SEED:?Set the seed already declared in genai_conditions.json}"
cd "$LAB_ROOT/code"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
exec "$VACE_ENV/bin/python" student_files/vace/infer.py run \
    --weights "$VACE_WEIGHTS" \
    --video "$VACE_VIDEO" \
    --artwork "$VACE_ARTWORK" \
    --masks "$VACE_MASKS" \
    --seed "$VACE_SEED" \
    --output outputs/genai/output.mp4 \
    "$@"

