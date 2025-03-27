#!/bin/bash
set -x

MODELS=("gemini-2.0-flash-001")
# MODELS=("gpt-4o-mini")
# MODELS=("gemini-2.0-flash-001" "gpt-4o-mini")
SCENARIOS=("--prompt base" "--prompt minimal" "--prompt base --perturbation orthographic" "--prompt base --perturbation llm"
"--prompt base --perturbation L2" "--prompt base --perturbation LazyUser" "--prompt base --perturbation lexicalphrasal" "--prompt base --perturbation register" "--prompt base --perturbation typos_synthetic")
# SCENARIOS=("--prompt base" "--prompt minimal")

# SPLIT="micro_test"
SPLIT="test"

# Output: '../output_translations/wmt24/system-outputs/{model.short}/three/{split}/noising_{perturbation}_{prompt_id}_{split}_results.jsonl'
for model in "${MODELS[@]}"; do
  for scenario in "${SCENARIOS[@]}"; do
    python -m main --model "${model}" ${scenario} --split $SPLIT --mem_percent 0.9 --gpus 2
  done
done


# Evaluation
# data/translated/ points to experiments/output_translations/wmt24/system-outputs/
# python evaluation/03-eval_metrics.py ../../data/translated/gemini/three/$SPLIT/*.jsonl

# sbatch -J closed-models-tiny.out -p gpu-troja,gpu-ms --mem=10G --exclude=tdll-8gpu1 --constraint=gpuram48G --gpus=1 -n 1 -N 1 -c 1 run_experiments_closed.sh
# sbatch -J closed-models-test.out --mem=10G -n 1 -N 1 -c 1 run_experiments_closed.sh
