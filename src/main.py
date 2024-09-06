#%%
import os
import json
import pandas as pd

from models import sampling_params, load_model
from utils import *

data_folder = "../mt-metrics-eval-v2/wmt23/sources"
out_folder = "../new_translations/wmt23/system-outputs"
ref_suffix = ".refA.txt"
ref_folder = "../mt-metrics-eval-v2/wmt23/references"

with open('../mt_base.json', 'r') as file:
    prompts = json.load(file)

languages = ["de-en", "cs-uk"] #,"en-zh"
sample = 500


#%%
def main():
    args = parse_arguments()

    llm, model_short = load_model(args.model, args.gpus)

    testset = load_sample(f'{data_folder}/{args.lp}.txt', sample)
    references = load_sample(f'{ref_folder}/{args.lp}{ref_suffix}', sample)

    for prompt in prompts:
        for i in range(0, 110, 10):
            model_inputs = []
            translations = []
            source_language = args.lp.split("-")[0]
            target_language = args.lp.split("-")[1]

            # Try the rest of the prompt variants
            #if "01-src" not in prompt['id']:
            #    continue

            template = prompt['prompt'].replace("[target language]", CODE_MAP[target_language]).replace("[source language]", CODE_MAP[source_language])
            experiment_name = f"{model_short}_{sample}_v2_" + prompt['id'] + f"_{i}"

            for item in testset:

                noised_template = make_typos(template, i/float(100), item)
                noised_template = CHAT_INTRO[model_short] + noised_template.replace('[source sentence]', item) + CHAT_OUTRO[model_short]

                model_inputs.extend([noised_template])


            print("Model inputs are built. Starting generation")
            # Generate translations
            outputs = llm.generate(model_inputs, sampling_params)
            translations = [o.outputs[0].text for o in outputs]

            # create folder translations/wmt23/system-outputs
            if not os.path.exists(f'../new_translations/wmt23/system-outputs/{args.lp}'):
                os.makedirs(f'../new_translations/wmt23/system-outputs/{args.lp}')
            dict = {'source': testset, 'translation': translations, 'reference': references}
            df = pd.DataFrame(dict)
            with open(f'../new_translations/wmt23/system-outputs/{args.lp}/{experiment_name}.csv', 'w') as f:
                df.to_csv(f)
            
# %%

if __name__ == "__main__":
    main()

# usage:
# python -m main --lp cs-uk --model Unbabel/TowerInstruct-7B-v0.2

### working models:
# Unbabel/TowerInstruct-7B-v0.2
# Unbabel/EuroLLM-1.7B-Instruct
# meta-llama/Meta-Llama-3.1-8B-Instruct
