#!/usr/bin/env bash
# Prepare dependencies only; no weights or generation. [AI-CODE]
set -euo pipefail
: "${VACE_ENV:?Set VACE_ENV to an environment directory on cluster storage}"
VACE_PYTHON="${VACE_PYTHON:-python3.11}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$VACE_ENV/pyvenv.cfg" ]]; then
    "$VACE_PYTHON" -m venv "$VACE_ENV"
fi
"$VACE_ENV/bin/python" -m pip install --upgrade 'pip>=24,<26'
"$VACE_ENV/bin/python" -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu126
"$VACE_ENV/bin/python" -m pip install -r "$script_dir/requirements.txt"
"$VACE_ENV/bin/python" -m pip check
"$VACE_ENV/bin/python" -m pip freeze > "$VACE_ENV/installed-packages.txt"
printf 'Environment ready: %s\nWeights download is a separate command.\n' "$VACE_ENV"

