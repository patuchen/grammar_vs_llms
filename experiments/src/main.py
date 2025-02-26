#%%
import os
import json
import pandas as pd
import sys
from models import Model, default_sampling_params, load_model
from utils import *

import grammar_v_mtllm
from argparse import Namespace


#%%
def main(args=None):
    if args is None:
        args = parse_arguments()

    # load data
    data = grammar_v_mtllm.utils.load_data(split=args.split, langs=f"{args.lp}")
    # load model
    model = load_model(args.model, args.gpus, args.mem_percent)

    if not args.perturbation:
        with open(f'../prompts/mt_{args.prompt}.json', 'r') as file:
            prompts = json.load(file)

    else:
        with open(f'../noised_prompts/mt_{args.prompt}_noised_{args.perturbation}.json', 'r') as file:
            prompts = json.load(file)

    os.makedirs(f'../output_translations/wmt24/system-outputs/{model.short}/{args.lp}', exist_ok=True)
    # run experiments
    for prompt in prompts:
        data_translated = [{} for _ in data]
        prompt_text = prompt['prompt'] if not args.perturbation else prompt['noised_prompt']
        prompt_id = prompt.get("prompt_id") if not args.perturbation else prompt.get("noised_prompt_id")

        model_inputs = []
        for item in data:
            # print(item)
            model_inputs.extend([prompt_text.replace("{source_lang}", CODE_MAP[item['langs'][:2]]).replace("{target_lang}", CODE_MAP[item['langs'][3:]]).replace("{source_text}", item['src'])])
        print("Model inputs are built. Starting generation")

        # generate translations
        translations = model.generate(model_inputs)

        # save translations
        for idx, translation in enumerate(translations):
            data_translated[idx]["src"] = data[idx]["src"]
            data_translated[idx]["ref"] = data[idx]["ref"]
            data_translated[idx]["langs"] = data[idx]["langs"]
            data_translated[idx]['model'] = model.short
            data_translated[idx]['prompt_src'] = prompt.get("prompt_src")
            data_translated[idx]['model_input'] = model_inputs[idx]
            data_translated[idx]['prompt'] = prompt_text
            data_translated[idx]['prompt_p'] = prompt.get("prompt_p", None)
            data_translated[idx]['prompt_noiser'] = prompt.get("prompt_noiser", None)
            data_translated[idx]['tgt'] = translation

        # save data as jsonl
        with open(f'../output_translations/wmt24/system-outputs/{model.short}/{args.lp}/{prompt_id}_{args.split}_results.jsonl', 'w', encoding='utf-8') as f:
            for item in data_translated:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
# %%

if __name__ == "__main__":

    
    # if command line arguments are provided, use those
    if len(sys.argv) > 1:
        args = parse_arguments()

    else:
        # Otherwise, use these hard-coded values (or modify as needed)
        args = Namespace(
            lp='cs-uk',
            model='Unbabel/TowerInstruct-7B-v0.2',
            prompt='base',
            split='micro_test',
            gpus=1,
            perturbation="character_noise"
        )

    main(args)


# usage:
# python -m main --lp cs-uk --model Unbabel/TowerInstruct-7B-v0.2 --prompt base --split micro_test

### working models:
# Unbabel/TowerInstruct-7B-v0.2
# Unbabel/EuroLLM-1.7B-Instruct
# meta-llama/Meta-Llama-3.1-8B-Instruct
# gpt-4o-mini
# claude-3-5-haiku-latest
