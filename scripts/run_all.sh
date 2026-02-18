#!/usr/bin/env bash
# Train all 4 models in parallel, each on its own GPU.
set -e

cd "$(dirname "$0")/.."
mkdir -p outputs/{proteinmpnn,alphafold2,rfdiffusion,esm2}

echo "Launching all 4 models in parallel..."
python scripts/train_and_eval.py --model proteinmpnn   --gpu 0 2>&1 | tee outputs/proteinmpnn/console.log &
python scripts/train_and_eval.py --model alphafold2    --gpu 1 2>&1 | tee outputs/alphafold2/console.log &
python scripts/train_and_eval.py --model rfdiffusion   --gpu 2 2>&1 | tee outputs/rfdiffusion/console.log &
python scripts/train_and_eval.py --model esm2          --gpu 3 2>&1 | tee outputs/esm2/console.log &

echo "Waiting for all models to finish..."
wait
echo "All training and evaluation complete!"
