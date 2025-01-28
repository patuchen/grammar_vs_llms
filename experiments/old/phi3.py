#!/usr/bin/env python3
import ipdb
import random
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os
import json
from tqdm import tqdm
import sys

data_folder = "mt-metrics-eval-v2/wmt23/sources/"
out_folder = "new_translations/wmt23/system-outputs"
ref_suffix = ".refA.txt"
ref_folder = "mt-metrics-eval-v2/wmt23/references/"
mapping = {
    "cs": "Czech",
    "uk": "Ukrainian",
    "de": "German",
    "en": "English",
    "he": "Hebrew",
    "ru": "Russian",
    "zh": "Chinese",
}

languages = ["de-en", "cs-uk", "en-zh"] #"de-en"
sample = 500

# Kathy's prompt seems working
SYSTEM_PROMPT = "You are a machine translation system. Rules:\n" + \
        "1. Provide a precise and correct translation in the target language.\n" + \
        "2. Give the answer without quotes, and without explanations.\n" + \
        "3. Don't say anything else after the translation.\n"

TEMPLATE = """<|system|>
{system_prompt}<|end|>
<|user|>
{user_prompt}<|end|>
<|assistant|>

"""

def load_prompts(path = 'mt_base.json'):
    with open(path, 'r') as file:
        prompts = json.load(file)
    return prompts


def load_sample(path, sample):
    loaded = []
    with open(path, 'r') as f:
        i = 0
        for line in f:
            if i >= sample:
                #i += 1
                #continue
                break
            loaded.append(line.strip())
            i += 1
    return loaded

def make_typos(sentence, prob_threshold=0.1, seed=42):
    new_sentence = ""
    i = 0
    random.seed(seed)
    while i < len(sentence):
        # Do not replace anything in placeholders - there is still the source sentence placeholder
        if sentence[i] == '[':
            close_i = sentence.find(']', i)
            new_sentence += sentence[i:close_i+1]
            i = close_i + 1
            continue
        # With a random probability of prob_threshold, introduce a typo
        if random.Random().random() < prob_threshold:
            # Only swap if not approaching the end of the sentence and if the characters are not special
            if i+1 < len(sentence) and not re.match(r'\W', sentence[i]) and not re.match(r'\W', sentence[i+1]):
                new_sentence += sentence[i+1] + sentence[i] #Swap two consecutive letters
                i += 2
                continue
            # Not used ATM, only swapping. Or omit the current letter (i.e don't copy it to the new string)
        else:
            new_sentence += sentence[i]
        i += 1
    return new_sentence

def load_pipeline(device=0, flash_attention = False): # Some GPUs don't support it
    model = None
    if flash_attention:
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3.5-mini-instruct",
            device_map=f"cuda",                      
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3.5-mini-instruct",
            device_map=f"cuda",            
            torch_dtype="auto",
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"        
    )
    return pipe

def run(device, lp, prompt, i):
    # prompts = load_prompts()
    pipe = load_pipeline(device)

    # for lp in languages:
    print("Langage:", lp)
    # load testset
    testset = load_sample(f'{data_folder}/{lp}.txt', sample)
    references = load_sample(f'{ref_folder}{lp}{ref_suffix}', sample)
    
    # for prompt in prompts:
        # for i in range(0, 110, 10):
    translations = []
    source_language = lp.split("-")[0]
    target_language = lp.split("-")[1]
    template = prompt['prompt'].replace("[target language]", mapping[target_language]).replace("[source language]", mapping[source_language])
    experiment_name = f"{sample}_v2" + prompt['id'] + f"_{i}"


    msgs = []
    for item in testset:
        noised_template = make_typos(template, i/float(100), item)
        noised_template = noised_template.replace('[source sentence]', item)
        msgs.append(TEMPLATE.format(system_prompt=SYSTEM_PROMPT, user_prompt=noised_template))

    
    generation_args = { 
        "max_new_tokens": 500, 
        "return_full_text": False, 
        "temperature": None, 
        "do_sample": False, 
    } 
    translations = pipe(msgs, **generation_args)
    translations = [i[0]["generated_text"] for i in translations]
    # create folder translations/wmt23/system-outputs
    if not os.path.exists(f'{out_folder}/{lp}'):
        os.makedirs(f'{out_folder}/{lp}')
    dict = {'source': testset, 'translation': translations, 'reference': references}
    df = pd.DataFrame(dict)
    with open(f'{out_folder}/{lp}/{experiment_name}.csv', 'w') as f:
        df.to_csv(f)

if __name__ == "__main__":
    # This part assumes that we are working with SLURM job arrays
    task = os.environ.get("SLURM_ARRAY_TASK_ID")

    # TASKS: language * prompts * 11 = 3 * 4 * 10 = 132
    param_space = []
    prompts = load_prompts()
    device = 0
    for lp in languages: # 3
        for prompt in prompts: # 4
            for i in range(0, 110, 10): # 11
                param_space.append(
                    {
                        "device": device,
                        "lp": lp,
                        "prompt": prompt,
                        "i": i,
                    }
                )
                device += 1
                device %= 6 # 6 GPUS
    params = param_space[int(task)-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params["device"])
    run(**params)
