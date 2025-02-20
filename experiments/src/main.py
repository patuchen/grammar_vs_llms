#%%
import os
import json
import pandas as pd

from models import Model, sampling_params
from utils import *

import grammar_v_mtllm
from argparse import Namespace


#%%
def main(args=None):
    if args is None:
        args = parse_arguments()

    # load model
    model = Model(args.model, args.gpus)

    # load data
    data = grammar_v_mtllm.utils.load_data(split=args.split, langs=f"{args.lp}")

    # load prompts
    with open(f'../prompts/mt_{args.prompt}.json', 'r') as file:
        prompts = json.load(file)

    # set noising function
    noising_functions = {
        "character_noise": make_typos,
    }
    noising_function = noising_functions[args.perturbation]

    # set languages
    source_language = args.lp.split("-")[0]
    target_language = args.lp.split("-")[1]

    # run experiments
    for prompt in prompts:

        # for synthetic noise, repeat at increasing noise levels
        if args.perturbation == "character_noise":
            for i in range(0, 110, 10):
                model_inputs = []

                template = prompt['prompt'].replace("[target language]", CODE_MAP[target_language]).replace("[source language]", CODE_MAP[source_language])
                experiment_name = f"{model.short}_{args.split}_v2_" + prompt['id'] + f"_{i}"

                # build model inputs
                for item in data:
                    # use the source sentence as seed
                    noised_template = noising_function(template, i/float(100), item["src"])

                    # add system prompt
                    noised_template = CHAT_INTRO[model.short] + noised_template.replace('[source sentence]', item["src"]) + CHAT_OUTRO[model.short]

                    model_inputs.extend([noised_template])

                print("Model inputs are built. Starting generation")

                # generate translations
                translations = model.generate(model_inputs)

                # save translations
                for idx, translation in enumerate(translations):
                    data[idx]["tgts"].append({"tgt": translation, "model": model.short, "prompt": prompt['id'], "perturbation": f"{i/float(100)},character_noise", "lp": args.lp})
                break

        # for non-synthetic noise, generate translations for different prompts
        else:
            model_inputs = []
            for item in data:
                model_inputs.extend([prompt['prompt'].replace("[target language]", CODE_MAP[target_language]).replace("[source language]", CODE_MAP[source_language])])
            print("Model inputs are built. Starting generation")

            # generate translations
            translations = model.generate(model_inputs)

            # save translations
            for idx, translation in enumerate(translations):
                data[idx]["tgts"].append({"tgt": translation, "model": model.short, "prompt": prompt['id'], "perturbation": "NA", "lp": args.lp})

        break

    if not os.path.exists(f'../output_translations/wmt24/system-outputs/{args.lp}'):
        os.makedirs(f'../output_translations/wmt24/system-outputs/{args.lp}')

    # save data as jsonl
    with open(f'../output_translations/wmt24/system-outputs/{args.lp}/{args.prompt}_{args.split}_results.jsonl', 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
# %%

if __name__ == "__main__":

    # either define args here or pass them as arguments on command line
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
