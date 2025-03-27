#!/bin/bash

set -x

MODELS=("utter-project/EuroLLM-9B-Instruct") # "Unbabel/TowerInstruct-7B-v0.2")
SCENARIOS=("--prompt base" "--prompt minimal" "--prompt base --perturbation orthographic" "--prompt base --perturbation typos_synthetic" "--prompt base --perturbation L2" "--prompt base --perturbation LazyUser" "--prompt base --perturbation llm" "--prompt base --perturbation lexicalphrasal" "--prompt base --perturbation register")

for model in "${MODELS[@]}"; do
  for scenario in "${SCENARIOS[@]}"; do
    python -m main --model "${model}" ${scenario} --split test --mem_percent 0.9 --gpus 1
  done
done
