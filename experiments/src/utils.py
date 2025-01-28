import random
import re
import argparse
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

# LLAMA3_TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + \
# "{{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>" + \
# "{{ user_msg_1 }}"

# CHAT_INTRO = {"Meta":dedent("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
#             You are a helpful machine translation assistant.
#             <|eot_id|><|start_header_id|>user<|end_header_id|>"""),
#             "EuroLLM": dedent("""<|im_start|>system
#             You are a helpful machine translation assistant.
#             <|im_end|>
#             <|im_start|>user
#             """),
#             "TowerInstruct": dedent("""<|im_start|>system
#             You are a helpful machine translation assistant.
#             <|im_end|>
#             <|im_start|>user
#             """)
#             }

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
            """)
}


CHAT_OUTRO = {
    "Meta": dedent("""<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""),
    "EuroLLM": dedent("""<|im_end|>
        <|im_start|>assistant
        """),
    "TowerInstruct": dedent("""<|im_end|>
        <|im_start|>assistant
        """)
    }

                

    
# CHAT_INTRO = """<|im_start|>system
# You are a helpful machine translation assistant.
# <|im_end|>
# <|im_start|>user
# """

# CHAT_OUTRO = {"Meta":dedent("""<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""),
# "EuroLLM": dedent(""" <|im_end|>
# <|im_start|>assistant
# """),
# "TowerInstruct": dedent(""" <|im_end|>
# <|im_start|>assistant
# """)
# }

#              """ <|im_end|>
# <|im_start|>assistant
# """
        

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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generates samples for MBR decoding using EuroLLM.")
    parser.add_argument(
        "--lp",
        type=str,
        required=True,
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
        "--num_candidates",
        type=int,
        default=1,
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