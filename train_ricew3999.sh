#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-50}"
WORKERS="${WORKERS:-4}"
VAL_EVERY="${VAL_EVERY:-1}"
TAG="${1:-ricew3999_ep${EPOCHS}_bs${BATCH_SIZE}}"
if [[ ! "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "Invalid experiment tag" >&2
  exit 1
fi
DATA_PATH="${DATA_PATH:-$PWD/data/RiceW3999}"
WEIGHTS_PATH="${WEIGHTS_PATH:-$PWD/pretrained/vgg16_bn-6c64b313.pth}"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "exp/$TAG/train.log"
EXTRA=()
if [[ "${AMP:-0}" == 1 ]]; then EXTRA+=(--amp); fi
"$PYTHON_BIN" -u train.py \
  --data-path "$DATA_PATH" --pretrained "$WEIGHTS_PATH" \
  --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" --workers "$WORKERS" \
  --eval-every "$VAL_EVERY" --tag "$TAG" --auto-resume "${EXTRA[@]}" \
  2>&1 | tee -a "exp/$TAG/train.log/running.log"
