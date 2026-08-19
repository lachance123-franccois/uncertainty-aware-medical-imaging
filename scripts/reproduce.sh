#!/usr/bin/env bash
# Reproduce every experiment in the report.
# Usage:  bash scripts/reproduce.sh [OUTPUT_DIR]
set -euo pipefail

OUT=${1:-runs}
EPOCHS=${EPOCHS:-25}
MEMBERS=${MEMBERS:-5}
SAMPLES=${SAMPLES:-30}

for DATASET in pneumoniamnist breastmnist; do
  echo "=============================================================="
  echo " $DATASET  (epochs=$EPOCHS, ensemble=$MEMBERS, mc-samples=$SAMPLES)"
  echo "=============================================================="
  python -m umi.cli run-all \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --n-members "$MEMBERS" \
    --n-samples "$SAMPLES" \
    --corruption gaussian_noise \
    --severities 1 2 3 4 5 \
    --out "$OUT/$DATASET"
done

# Seed sensitivity: three independent repetitions of the whole pipeline.
for SEED in 0 1 2; do
  python -m umi.cli run-all --dataset pneumoniamnist --epochs "$EPOCHS" \
    --n-members "$MEMBERS" --seed "$SEED" --no-maps --out "$OUT/seed_$SEED"
done

echo "Done. Reports: $OUT/*/REPORT.md"
