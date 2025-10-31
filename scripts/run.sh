#!/usr/bin/env bash#!/usr/bin/env bash

# Minimal run script: usage ./scripts/run.sh data.csv# Minimal run script: usage ./scripts/run.sh data.csv

set -euo pipefailset -euo pipefail

DATA_PATH="$1"DATA_PATH="$1"

python -m src.smoking_cessation_ml.main --data "$DATA_PATH"python -m src.smoking_cessation_ml.main --data "$DATA_PATH"

