#!/bin/bash

set -x

MODELS=("utter-project/EuroLLM-9B-Instruct" "Unbabel/TowerInstruct-7B-v0.2")
SCENARIOS=("--prompt base" "--prompt minimal" "--prompt base --perturbation orthographic_over_p")

for model in "${MODELS[@]}"; do
  for scenario in "${SCENARIOS[@]}"; do
    python -m main --model "${model}" ${scenario} --split micro_test --mem_percent 0.9 --gpus 2
  done
done
