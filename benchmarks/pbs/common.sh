#!/bin/bash
# Shared setup for the hrrr_fast benchmark jobs on Derecho.
#
# Why a shared file: every benchmark must run against the same code checkout, the same
# Python environment and the same exported weights, otherwise timings are not comparable.
set -euo pipefail
export PTYPE_FAST=/glade/work/dgagne/ptype_fast
export REPO=${REPO:-$PTYPE_FAST/ptype-physical}
export WEIGHTS=$PTYPE_FAST/weights
export RESULTS=${RESULTS:-$REPO/benchmarks/results/derecho}
export SCRATCH_DIR=/glade/derecho/scratch/$USER/ptype_fast
# Keras is only needed for the baseline engine; the torch backend avoids a TensorFlow install.
export KERAS_BACKEND=torch
export PYTHONPATH=$REPO
# numba caches compiled kernels next to the source by default; a per-job scratch cache avoids
# stale-cache problems when the code changes between runs and keeps the repo clean.
export NUMBA_CACHE_DIR=$SCRATCH_DIR/numba_cache/${PBS_JOBID:-interactive}
mkdir -p "$RESULTS" "$NUMBA_CACHE_DIR" "$WEIGHTS"
# Casper has no system Python 3.12, so the Derecho environment does not load there; Casper jobs set VENV to
# an environment built on a uv-managed Python with the same package versions (docs/realtime.md).
source ${VENV:-$PTYPE_FAST/venv}/bin/activate
cd "$REPO"
echo "host=$(hostname) job=${PBS_JOBID:-none} commit=$(git rev-parse --short HEAD) date=$(date -u +%FT%TZ)"
if [ ! -f "$WEIGHTS/ptype_weights.npz" ]; then
    python applications/export_ptype_weights.py --model $PTYPE_FAST/model/ptype_model_20240909.keras \
        --scaler $PTYPE_FAST/model/ptype_scaler_20240909.json --out-dir "$WEIGHTS"
fi
