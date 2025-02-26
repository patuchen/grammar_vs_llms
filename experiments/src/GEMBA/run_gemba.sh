#!/bin/bash

python -m main --source=/home/saycock/personal/grammar_vs_llms/data/mt-metrics-eval-v2/wmt24/sources/cs-uk.txt \
    --hypothesis=/home/saycock/personal/grammar_vs_llms/data/mt-metrics-eval-v2/wmt24/system-outputs/cs-uk/GPT-4.txt \
    --source_lang=English \
    --target_lang=Czech \
    --method="GEMBA-DA" \
    --model="gpt-4o-mini" \
    --subset="tiny"

    # --hypothesis=/home/saycock/personal/grammar_vs_llms/data/mt-metrics-eval-v2/wmt24/references/cs-uk.refA.txt \
