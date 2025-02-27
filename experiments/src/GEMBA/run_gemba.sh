#!/bin/bash

#SBATCH --job-name=gemba
#SBATCH --output=jobs/slurm-%j.out
#SBATCH --mem=100G
#SBATCH --error=jobs/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8


python -m test \
        --lp en-de \
        --base_prompt GEMBA-DA \
        --model gpt-4o-mini \
        --subset tiny_test

python -m test \
        --lp cs-uk \
        --base_prompt GEMBA-DA \
        --model gpt-4o-mini \
        --subset tiny_test

python -m test \
        --lp en-zh \
        --base_prompt GEMBA-DA \
        --model gpt-4o-mini \
        --subset tiny_test

# python -m main --source=/home/saycock/personal/grammar_vs_llms/data/mt-metrics-eval-v2/wmt24/sources/cs-uk.txt \
#     --hypothesis=/home/saycock/personal/grammar_vs_llms/data/mt-metrics-eval-v2/wmt24/system-outputs/cs-uk/GPT-4.txt \
#     --source_lang=English \
#     --target_lang=Czech \
#     --method="GEMBA-DA" \
#     --model="gpt-4o-mini" \
#     --subset="tiny"

    # --hypothesis=/home/saycock/personal/grammar_vs_llms/data/mt-metrics-eval-v2/wmt24/references/cs-uk.refA.txt \

# python -m test --lp en-de --base_prompt GEMBA-DA --model gpt-4o-mini --subset micro_test