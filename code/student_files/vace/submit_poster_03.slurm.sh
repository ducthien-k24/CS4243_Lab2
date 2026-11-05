#!/usr/bin/env bash
# Fixed poster_03/ad4 segmented VACE job. Submit from the repository root.
# Select the actual NUS GPU partition/account with sbatch options. [AI-CODE]
#SBATCH --job-name=vace-poster03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=vace-poster03-%j.log

set -euo pipefail

resume_args=()
if (( $# > 1 )); then
    printf 'Usage: sbatch [scheduler options] submit_poster_03.slurm.sh [--resume]\n' >&2
    exit 2
fi
if (( $# == 1 )); then
    if [[ "$1" != "--resume" ]]; then
        printf 'Only --resume is accepted; model settings are fixed by the declared profile.\n' >&2
        exit 2
    fi
    resume_args=(--resume)
fi

: "${SLURM_JOB_ID:?Submit this file with sbatch on a GPU partition}"
LAB_ROOT="${LAB_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or export LAB_ROOT}}"
: "${VACE_ENV:?Export VACE_ENV to the existing VACE environment before sbatch}"
: "${VACE_WEIGHTS:?Export VACE_WEIGHTS to the downloaded Diffusers checkpoint before sbatch}"

if [[ "$LAB_ROOT" != /* || "$VACE_ENV" != /* || "$VACE_WEIGHTS" != /* ]]; then
    printf 'LAB_ROOT, VACE_ENV and VACE_WEIGHTS must be absolute cluster paths.\n' >&2
    exit 2
fi
cd "$LAB_ROOT/code"
python_bin="$VACE_ENV/bin/python"
runner="$LAB_ROOT/code/student_files/vace/segmented_infer.py"
if [[ ! -x "$python_bin" || ! -f "$runner" ]]; then
    printf 'Cannot find the VACE Python or runner. Check VACE_ENV and LAB_ROOT.\n' >&2
    exit 2
fi
for executable in ffmpeg ffprobe srun; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        printf 'Missing %s on PATH; load the required cluster module before sbatch.\n' "$executable" >&2
        exit 2
    fi
done

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mask_dir="$LAB_ROOT/code/outputs/genai/inputs/masks"
shopt -s nullglob
board_masks=("$mask_dir"/mask_*.png)
if (( ${#board_masks[@]} != 450 )); then
    printf 'Expected 450 fresh poster_03 board masks in %s; found %s. Export Notebook 4 inputs first.\n' \
        "$mask_dir" "${#board_masks[@]}" >&2
    exit 2
fi
unset board_masks

args=(
    --video "$LAB_ROOT/data/personal/poster_03/clip.mp4"
    --artwork "$LAB_ROOT/data/artwork/ad4.png"
    --masks "$mask_dir"
    --weights "$VACE_WEIGHTS"
    --conditions "$LAB_ROOT/code/manifests/genai_conditions.json"
    --log "$LAB_ROOT/code/manifests/genai_log.csv"
    --seed 2026
    --width 832 --height 480 --fps 16 --num-frames 81
    --steps 50 --guidance 5 --conditioning-scale 1
    --flow-shift 3 --offload sequential
    --output "$LAB_ROOT/code/outputs/genai/output.mp4"
)

printf 'Job %s | host %s | repo %s\n' "$SLURM_JOB_ID" "${HOSTNAME:-unknown}" "$LAB_ROOT"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
fi
printf 'Checking inputs; no model is loaded during preflight.\n'
"$python_bin" "$runner" preflight "${args[@]}"
printf 'Starting one declared attempt: three sequential 5-second segments.\n'
exec srun --ntasks=1 "$python_bin" "$runner" run "${args[@]}" "${resume_args[@]}"
