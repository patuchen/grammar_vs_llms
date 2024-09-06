#%%
import os
import argparse
import json

data_folder = "mt-metrics-eval-v2/wmt23/sources/"
out_folder = "translations/wmt23/system-outputs"
ref_suffix = ".refA.txt"
ref_folder = "mt-metrics-eval-v2/wmt23/references/"
CODE_MAP = {
    "uk": "Ukrainian",
    "cs": "Czech",
    "de": "German",
    "es": "Spanish (Latin America)",
    "hi": "Hindi",
    "is": "Icelandic",
    "ja": "Japanese",
    "ru": "Russian",
    "zh": "Chinese",
    "en": "English",
}

with open('mt_prompts.json', 'r') as file:
    prompts = json.load(file)

# %%

languages = ["de-en", "cs-uk", "en-zh"] #,
sample = 500

import random
import re

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


#%%

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

#%%
import argparse
import os
import pandas as pd
from vllm import LLM, SamplingParams
CHATML_TEMPLATE = """<|im_start|>system
You are a professional {source_language} to {target_language} translator. Your goal is to accurately convey the meaning and nuances of the original {source_language} text while adhering to {target_language} grammar, vocabulary, and cultural sensitivities.
<|im_end|>
<|im_start|>user
Translate the following {source_language} source text to {target_language}:
{source_language}: {source}
{target_language}: <|im_end|>
<|im_start|>assistant
"""

#%%

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generates samples for MBR decoding using EuroLLM.")
    parser.add_argument(
        "--lp",
        type=str,
        required=True,
        help="Language pair code from WMT24.",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=20,
        help="Number of samples to generate for each source sentence.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs used to generate the candidates (default: 1).",
    )
    parser.add_argument(
        "--output_file", type=str, help="Output file with translation candidates."
    )
    return parser.parse_args()



#%%
def main():
    args = parse_arguments()
    sampling_params = SamplingParams(
        use_beam_search=False,
        best_of=1,
        # Since we are doing sentence level we can stop at newlines. This avoids the model trying to continue
        # generating translations
        stop=["<|im_end|>", "\n"], 
        # epsilon sampling: min_p=0.02
        # min_p=0.02,

        # nucleus sampling: top_p=0.9
        # top_p=0.9,

        # greedy: temp=0
        temperature=0.05,

        max_tokens=512,
    )

    llm = LLM(model="utter-project/EuroLLM-1.7B-Instruct", tensor_parallel_size=1)
    source_sentences = [s.strip() for s in open(f"data/sources/{args.lp}.txt").readlines()]

    model_inputs = []
    for source in source_sentences:
        # Build MBR Inputs!
        source_language = CODE_MAP[args.lp.split("-")[0]]
        target_language = CODE_MAP[args.lp.split("-")[1]]

        source_prompt = CHATML_TEMPLATE.format(
            source_language=source_language,
            target_language=target_language,
            source=source,
        )
        # Each prompt will appear several times. VLLM also performs some prompt caching thus
        # this is actually fast.
        model_inputs.extend([source_prompt] * args.num_candidates)

    print("Model inputs are built. Starting generation")
    # Generate translations
    outputs = llm.generate(model_inputs, sampling_params)
    generations = [o.outputs[0].text for o in outputs]
    # Extract the directory from the output_file path
    output_directory = os.path.dirname(args.output_file)
    # Create the directory if it does not exist
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    print (f"Saving translations to {args.output_file}")
    with open(args.output_file, 'w') as file:
        for line in generations:
            file.write(line + '\n')


if __name__ == "__main__":
    main()




# from transformers import AutoTokenizer, AutoModelForCausalLM

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="auto", quantization_config={"load_in_8bit": True})