#!/bin/bash
#SBATCH --job-name=nano-perf-repair
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=perf-repair-%j.log
set -euo pipefail
# Submit with the site's EXISTING RTX5090 partition/node/account flags. Do not
# assume an empty queue from SSH failure. Activate the existing venv BEFORE sbatch.
: "${TARGET_MODEL:?set TARGET_MODEL to existing Qwen3-8B path}"
: "${DRAFT_MODEL:?set DRAFT_MODEL to existing Qwen3-0.6B path}"
: "${FROZEN_DEBUG_INPUTS:?set strict JSON described in README}"
: "${REFERENCE_ROOT:?set unmodified extracted attachment reference root}"
: "${REPAIR_RESULTS:?set a NEW output directory}"
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
mkdir -p "$REPAIR_RESULTS"
python - <<'PY'
import torch,flash_attn
assert torch.cuda.is_available()
assert torch.__version__.startswith('2.9.1'),torch.__version__
assert flash_attn.__version__.split('+')[0]=='2.8.3',flash_attn.__version__
print('ENV',torch.__version__,torch.version.cuda,flash_attn.__version__,torch.cuda.get_device_name(0))
PY
# No environment reconstruction or dependency installation inside the allocation.
PERF_REPAIR_REQUIRE_CUDA=1 python -m pytest -q tests/perf_repair
python tools/perf_repair_gpu_gate.py --target "$TARGET_MODEL" --draft "$DRAFT_MODEL" \
  --output "$REPAIR_RESULTS/gpu-gate.json"
# Only after the entire real gate passes. No old216 rerun, no confirmation set.
python tools/perf_repair_compare.py --reference-root "$REFERENCE_ROOT" \
  --target "$TARGET_MODEL" --draft "$DRAFT_MODEL" --inputs "$FROZEN_DEBUG_INPUTS" \
  --output-dir "$REPAIR_RESULTS/comparison" --seed 17011 --cap 512 \
  --repeats 2 --max-seconds 1800 --worker-seconds 300
