#!/usr/bin/env bash
#
# Set up the tunix venv on a TPU VM that already has python3.12 installed.
#   - Creates venv at ~/venvs/tunix
#   - Installs the tunix / jax / flax stack
#
# Prereq: python3.12 must already be on PATH (or findable via `uv python
# find 3.12`). See tpu-setup.md for the one-time install step.
#
# Secrets (~/.env) and shell wiring (~/.bashrc) are intentionally NOT
# handled here — do those once, by hand, per tpu-setup.md.
#
# Usage:
#   git clone https://github.com/borisbolliet/tpu-2026.git
#   cd tpu-2026 && ./bootstrap.sh
#
# Idempotent — safe to re-run.

set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV=${VENV:-$HOME/venvs/tunix}

echo "==> Locating python3.12"
export PATH="$HOME/.local/bin:$PATH"
if command -v python3.12 >/dev/null 2>&1; then
  PYTHON312=$(command -v python3.12)
elif command -v uv >/dev/null 2>&1 && PYTHON312=$(uv python find 3.12 2>/dev/null); then
  :
else
  echo "ERROR: python3.12 not found on PATH and uv can't locate one." >&2
  echo "       Install it first (see tpu-setup.md), e.g.:" >&2
  echo "         curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  echo "         export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
  echo "         uv python install 3.12" >&2
  exit 1
fi
echo "    using $PYTHON312"

if [[ ! -d "$VENV" ]]; then
  echo "==> Creating venv at $VENV"
  "$PYTHON312" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel

echo "==> Installing pinned deps from requirements.txt"
# requirements.txt includes libtpu — without it, jax silently falls back to CPU
# on a TPU VM ("A Google TPU may be present ... Falling back to cpu").
pip install -r "$REPO_DIR/requirements.txt"

echo "==> Installing jax / tunix / qwix / flax from GitHub HEAD"
# Order matters: tunix pulls flax from PyPI, so we replace flax last.
pip install git+https://github.com/jax-ml/jax
pip install git+https://github.com/google/tunix git+https://github.com/google/qwix
pip uninstall -y flax
pip install git+https://github.com/google/flax

echo "==> Registering Jupyter kernel 'tunix'"
python -m ipykernel install --user --name tunix --display-name "tunix"

echo "==> Done. Activate the venv with:  source $VENV/bin/activate"
