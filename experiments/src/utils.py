import json
import random
import re
import argparse
from collections import defaultdict
from textwrap import dedent

CODE_MAP = {
    "cs": "Czech",
    "uk": "Ukrainian",
    "de": "German",
    "en": "English",
    "he": "Hebrew",
    "ru": "Russian",
    "zh": "Chinese",
}

CHAT_INTRO = {
    "Meta": dedent("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful machine translation assistant.
            <|eot_id|><|start_header_id|>user<|end_header_id|>"""),
    "EuroLLM": dedent("""<|im_start|>system
            You are a helpful machine translation assistant.
            <|im_end|>
            <|im_start|>user
            """),
    "TowerInstruct": dedent("""<|im_start|>system
            You are a helpful machine translation assistant.
            <|im_end|>
            <|im_start|>user
            """),
    "gpt": "",
}

CHAT_OUTRO = {
    "Meta": dedent("""<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""),
    "EuroLLM": dedent("""<|im_end|>
        <|im_start|>assistant
        """),
    "TowerInstruct": dedent("""<|im_end|>
        <|im_start|>assistant
        """),
    "gpt": "",
}


def load_prompts(args):
    if not args.perturbation:
        with open(f'../prompts/mt_{args.prompt}.json', 'r') as file:
            return json.load(file)

    else:
        with open(f'../noised_prompts/mt_{args.prompt}_noised_{args.perturbation}.json', 'r') as file:
            prompts = json.load(file)

    if args.perturbation == "llm":
        return prompts

    # sort into actual buckets
    bucketed_prompts = defaultdict(list)
    for prompt in prompts:
        if prompt["noised_prompt"] == prompt["prompt_src"]:  # happens in lexicalphrasal
            continue
        bucketed_prompts[prompt["prompt_id"] + prompt["prompt_noiser"]].append(prompt)
    # sample one prompt per bucket
    final_prompts = []
    for bucket in bucketed_prompts.values():
        final_prompts.append(random.sample(bucket, 1)[0])

    return prompts


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


# TODO: add other noising functions
def make_typos(sentence, prob_threshold=0.1, seed=42):
    new_sentence = ""
    i = 0
    random.seed(seed)
    while i < len(sentence):
        # Do not replace anything in placeholders - there is still the source sentence placeholder
        if sentence[i] == '[':
            close_i = sentence.find(']', i)
            new_sentence += sentence[i:close_i + 1]
            i = close_i + 1
            continue

        # With a random probability of prob_threshold, introduce a typo
        if random.Random().random() < prob_threshold:
            # Only swap if not approaching the end of the sentence and if the characters are not special
            if i + 1 < len(sentence) and not re.match(r'\W', sentence[i]) and not re.match(r'\W', sentence[i + 1]):
                new_sentence += sentence[i + 1] + sentence[i]  # Swap two consecutive letters
                i += 2
                continue
            # Not used ATM, only swapping. Or omit the current letter (i.e don't copy it to the new string)
        else:
            new_sentence += sentence[i]
        i += 1
    return new_sentence


def make_natural_typo(sentence, probs, seed=42):
    new_sentence = ""
    i = 0
    random.seed(seed)
    capital_letters = [ord('A') - ord('a')
                       if ord('A') <= ord(l) <= ord('Z') else 0 for l in sentence]
    sentence = sentence.lower()

    while i < len(sentence):
        # Do not replace anything in placeholders - there is still the source sentence placeholder
        if sentence[i] == '[':
            close_i = sentence.find(']', i)
            new_sentence += sentence[i:close_i + 1]
            i = close_i + 1
            continue

        if random.Random().random() < probs.get(sentence[i], 0):
            # Replace with a random nearby character
            # Neighbours dict is defined at the bottom
            new_sentence += chr(ord(random.choice(neighbours[sentence[i]])) + capital_letters[i])
        else:
            new_sentence += sentence[i]
        i += 1

    return new_sentence


def extract_translation(text):
    translation = text.strip()
    return translation


def parse_arguments():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--mem_percent",
        type=float,
        default=0.9,
        help="Percentage of memory to use for vLLM engine (default: 0.9).",
    )
    parser.add_argument(
        "--lp",
        type=str,
        default="three",
        help="Language pair code from WMT24.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="utter-project/EuroLLM-1.7B-Instruct",
        help="Model for generation.",
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
    parser.add_argument(
        "--prompt",
        type=str,
        default="base",
        choices=["base", "minimal"],
        help="'Pristine' prompt to use for generation.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="micro_test",
        help="Split to use for generation.",
    )
    parser.add_argument(
        "--perturbation",
        type=str,
        default=None,
        choices=[None, "orthographic", "llm", "L2", "LazyUser", "lexicalphrasal", "register", "typos_synthetic"],
        help="Perturbation to use for generation.",
    )
    return parser.parse_args()


neighbours = {
    'a': ['q', 'w', 's', 'z'],
    'b': ['v', ' ', 'g', 'h', 'n'],
    'c': ['x', 'd', 'f', 'v', ' '],
    'd': ['e', 'r', 'f', 'c', 'x', 's'],
    'e': ['w', 'r', 'd', 's'],
    'f': ['d', 'r', 't', 'g', 'v', 'c'],
    'g': ['f', 't', 'y', 'h', 'b', 'v'],
    'h': ['g', 'y', 'u', 'j', 'n', 'b'],
    'i': ['u', '8', '9', 'o', 'k', 'j'],
    'j': ['h', 'u', 'i', 'k', 'm', 'n'],
    'k': ['j', 'i', 'o', 'l', ',', 'm'],
    'l': ['k', 'o', 'p'],
    'm': ['n', 'j', 'k', ',', ' '],
    'n': ['b', 'h', 'j', 'm', ' '],
    'o': ['i', 'p', 'l', 'k'],
    'p': ['o', 'l'],
    'q': ['w', 'a'],
    'r': ['e', 't', 'f', 'd'],
    's': ['a', 'w', 'e', 'd', 'x', 'z'],
    't': ['r', 'y', 'g', 'f'],
    'u': ['y', 'i', 'j', 'h'],
    'v': ['c', 'f', 'g', 'b', ' '],
    'w': ['q', 'e', 's', 'a'],
    'x': ['z', 's', 'd', 'c', ' '],
    'y': ['t', 'u', 'h', 'g'],
    'z': ['a', 's', 'x'],
}
