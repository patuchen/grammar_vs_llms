"""
Script used to generate samples for MBR decoding using VLLM and EuroLLM.

```bash
python generate_samples.py --lp en-es --num_candidates 20 --gpus 1 --output_file data/mbr_samples/en-es-samples.txt
```
"""
import argparse
import os
import json
import re
import random

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

from transformers import set_seed
set_seed(42)

LLAMA3_TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + \
"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>" + \
"{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
# "{{ model_answer_1 }}<|eot_id|>"
SYSTEM_PROMPT = "You are a helpful machine translation assistant."
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_sample(path, sample):
    loaded = []
    with open(path, 'r') as f:
        i = 0
        for line in f:
            if i >= sample:
                break
            loaded.append(line.strip())
            i += 1
    return loaded


def parse_arguments():
    parser = argparse.ArgumentParser(description="Translate a set of (noised) prompts.")
    parser.add_argument(
        "--lp",
        type=str,
        required=True,
        help="Language pair code from WMT23."
        )
    parser.add_argument("--base_prompt", type=str, required=True, help="which base prompt to use",
                        choices=["01", "02", "03", "04"])
    parser.add_argument("--noise_level", type=int, required=True,
                        choices=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    return parser.parse_args()

def noise(prompt, sentence, noise_level):
    prob_threshold = noise_level / 100
    noised_prompt = ""
    i = 0
    random.seed(sentence)
    while i < len(prompt):
        # Do not replace anything in placeholders - there is still the source sentence placeholder
        if prompt[i] == '[':
            close_i = prompt.find(']', i)
            noised_prompt += prompt[i:close_i+1]
            i = close_i + 1
            continue
        # With a random probability of prob_threshold, introduce a typo
        if random.Random().random() < prob_threshold:
            # Only swap if not approaching the end of the sentence and if the characters are not special
            if i+1 < len(prompt) and not re.match(r'\W', prompt[i]) and not re.match(r'\W', prompt[i+1]):
                noised_prompt += prompt[i+1] + prompt[i]  # Swap two consecutive letters
                i += 2
                continue
            # Not used ATM, only swapping. Or omit the current letter (i.e don't copy it to the new string)
        else:
            noised_prompt += prompt[i]
        i += 1
    return noised_prompt

def divide_list_into_chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def main():

    data_folder = "mt-metrics-eval-v2/wmt23/sources/"
    out_folder = "translations/wmt23/system-outputs"
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

    with open('mt_base.json', 'r') as file:
        # base file
        prompts = json.load(file)

    args = parse_arguments()

    sample = 500  # hard-coded sorry
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config={"load_in_8bit": True})

    lp = args.lp
    # load testset
    source_sentences = load_sample(f'{data_folder}/{lp}.txt', sample)
    references = load_sample(f'{ref_folder}{lp}{ref_suffix}', sample)

    prompt_id = f"mt-{args.base_prompt}-src-no_errors"
    base_prompt = None
    for prompt in prompts:
        if prompt["id"] == prompt_id:
            base_prompt = prompt["prompt"]
    if base_prompt is None:
        raise ValueError("Base Prompt not found!")

    model_inputs = []
    prompt_lengths = []
    for source in source_sentences:
        # Build MBR Inputs!
        source_language = args.lp.split("-")[0]
        target_language = args.lp.split("-")[1]

        user_msg = base_prompt.replace(
            "[target language]", mapping[target_language]).replace(
                "[source language]", mapping[source_language])
        if args.noise_level > 0:
            user_msg = noise(user_msg, source, args.noise_level)
        user_msg = user_msg.replace(
                    "[source sentence]", source)

        full_message = LLAMA3_TEMPLATE.format(system_prompt=SYSTEM_PROMPT, user_msg=user_msg)

        encoded = tokenizer(full_message, return_tensors="pt")
        prompt_length = encoded['input_ids'].shape[1]
        # print(full_message)
        model_inputs.append(full_message)
        prompt_lengths.append(prompt_length)

    # model_inputs = model_inputs[:12]
    # and there I THOUGHT I could do batching. it ended up messing with getting outputs for some reason??
    # dataloader = DataLoader(model_inputs, batch_size=1)
    # Generate translations

    generations = []
    for samples in tqdm(model_inputs, total=len(model_inputs)):
        encoded = tokenizer(samples, return_tensors="pt", padding=True)
        encoded["input_ids"].to("cuda")
        encoded["attention_mask"].to("cuda")
        output = model.generate(
            encoded["input_ids"].to("cuda"),
            attention_mask=encoded["attention_mask"].to("cuda"),
            max_new_tokens=256,
            num_beams=1,
            do_sample=False,
            temperature=None,
            top_p=None,
            stop_strings=[ "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>" ],
            tokenizer=tokenizer
        )
        output = output.to("cpu")
        generations += output

    # print(f"{len(prompt_lengths)}")
    # print(f"num outputs: {len(generations)}, example; {generations[0]}")
    translations = []
    for length, output in zip(prompt_lengths, generations):
        # print(f"length: {length}, output: {output[length:]}")
        translation = tokenizer.decode(output[length:], skip_special_tokens=True)
        translations.append(
            translation.replace('\n', ' ').replace('\t', ' ')
        )
        # print(translations[-1])

    experiment_name = f"llama3_noise_{args.noise_level}_base_prompt_{args.base_prompt}"
    # create folder translations/wmt23/system-outputs
    if not os.path.exists(f'translations/wmt23/system-outputs/{lp}'):
        os.makedirs(f'translations/wmt23/system-outputs/{lp}')
    # print(f"len sources {len(source_sentences)}, len translations {len(translations)}, len references {len(references)}")
    # only for debugging!
    # source_sentences = source_sentences[:12]
    # references = references[:12]
    dict = {'source': source_sentences, 'translation': translations, 'reference': references}
    df = pd.DataFrame(dict)
    with open(f'translations/wmt23/system-outputs/{lp}/{experiment_name}.csv', 'w') as f:
        df.to_csv(f)

if __name__ == "__main__":
    main()
