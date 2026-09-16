#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import torch
try:
    import mamba_ssm
except ImportError as exc:
    raise SystemExit("mamba_ssm is missing from this Docker environment") from exc
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible in this Docker environment")
print("torch:", torch.__version__)
print("mamba_ssm:", getattr(mamba_ssm, "__version__", "installed"))
print("gpu:", torch.cuda.get_device_name(0))
PY

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
