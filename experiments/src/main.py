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
    # print(data[0])

    # load model
    model = load_model(args.model, args.gpus, args.mem_percent)


    # load prompts
    with open(f'../prompts/mt_{args.prompt}.json', 'r') as file:
        prompts = json.load(file)

    # set noising function
    noising_functions = {
        "character_noise": make_typos,
    }
    # noising_function = noising_functions[args.perturbation]

    # set languages
    # source_language = args.lp.split("-")[0]
    # target_language = args.lp.split("-")[1]

    # run experiments
    for prompt in prompts:

        # for synthetic noise, repeat at increasing noise levels
        if args.perturbation == "character_noise":
            for i in range(0, 110, 10):
                model_inputs = []

                experiment_name = f"{model.short}_{args.split}_v2_" + prompt['id'] + f"_{i}"

                # build model inputs
                for item in data:
                    # print(item)
                    # use the source sentence as seed
                    template = prompt['prompt'].replace("{target language}", CODE_MAP[item['langs'][:2]]).replace("{source language}", CODE_MAP[item['langs'][3:]])
                    noised_template = noising_function(template, i/float(100), item["src"])

                    # add system prompt
                    # noised_template = CHAT_INTRO[model.short] + noised_template.replace('[source sentence]', item["src"]) + CHAT_OUTRO[model.short]

                    model_inputs.extend([noised_template])

                print("Model inputs are built. Starting generation")

                # generate translations
                translations = model.generate(model_inputs)

                # save translations
                for idx, translation in enumerate(translations):
                    if "tgts" not in data[idx]:
                        data[idx]["tgts"] = []
                    data[idx]["tgts"].append({"tgt": translation, "model": model.short, "prompt": prompt['id'], "perturbation": f"{i/float(100)},character_noise", "lp": args.lp})

        # for non-synthetic noise, generate translations for different prompts
        else:
            model_inputs = []
            for item in data:
                # print(item)
                model_inputs.extend([prompt['prompt'].replace("{target language}", CODE_MAP[item['langs'][:2]]).replace("{source language}", CODE_MAP[item['langs'][3:]])])
            print("Model inputs are built. Starting generation")

            # generate translations
            translations = model.generate(model_inputs)

            # save translations
            for idx, translation in enumerate(translations):
                if "tgts" not in data[idx]:
                    data[idx]["tgts"] = []
                data[idx]["tgts"].append({"tgt": translation, "model": model.short, "prompt_src": prompt["prompt_src"], "prompt": prompt['id'], "perturbation": "NA", "prompt_p": prompt.get("prompt_p", "None")})

    if not os.path.exists(f'../output_translations/wmt24/system-outputs/{args.lp}'):
        os.makedirs(f'../output_translations/wmt24/system-outputs/{args.lp}')

    # save data as jsonl
    with open(f'../output_translations/wmt24/system-outputs/{args.lp}/{args.prompt}_{args.split}_results.jsonl', 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
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
