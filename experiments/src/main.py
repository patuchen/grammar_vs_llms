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
    data_translated = [{} for _ in data]
    # print(data[0])

    # load model
    model = load_model(args.model, args.gpus, args.mem_percent)


    # load prompts
    with open(f'../prompts/mt_{args.prompt}.json', 'r') as file:
        prompts = json.load(file)

    # noising_function = noising_functions[args.perturbation]

    # run experiments
    for prompt in prompts:
        # for synthetic noise, repeat at increasing noise levels
                # for non-synthetic noise, generate translations for different prompts
        model_inputs = []
        for item in data:
            # print(item)
            model_inputs.extend([prompt['prompt'].replace("{target_lang}", CODE_MAP[item['langs'][:2]]).replace("{source_lang}", CODE_MAP[item['langs'][3:]]).replace("{source_line}", item['src'])])
        print("Model inputs are built. Starting generation")

        # generate translations
        translations = model.generate(model_inputs)

        # save translations
        for idx, translation in enumerate(translations):
            data_translated[idx]["src"] = data[idx]["src"]
            data_translated[idx]["ref"] = data[idx]["ref"]
            data_translated[idx]["langs"] = data[idx]["langs"]
            data_translated[idx]['model'] = model.short
            data_translated[idx]['prompt_src'] = prompt["prompt_src"]
            data_translated[idx]['model_input'] = model_inputs[idx]
            data_translated[idx]['prompt'] = prompt['prompt']
            data_translated[idx]['prompt_p'] = prompt.get("prompt_p", None)
            data_translated[idx]['prompt_noiser'] = prompt.get("prompt_noiser", None)
            data_translated[idx]['tgt'] = translation

#             if "tgts" not in data[idx]:
#                 data[idx]["tgts"] = []
#             data[idx]["tgts"].append({"tgt": translation, "model": model.short, "prompt_src": prompt["prompt_src"], "prompt": prompt['id'], "perturbation": "NA", "prompt_p": prompt.get("prompt_p", "None")})
# 
    if not os.path.exists(f'../output_translations/wmt24/system-outputs/{args.lp}'):
        os.makedirs(f'../output_translations/wmt24/system-outputs/{args.lp}')

    # save data as jsonl
    with open(f'../output_translations/wmt24/system-outputs/{args.lp}/{args.prompt}_{args.split}_results.jsonl', 'w', encoding='utf-8') as f:
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
