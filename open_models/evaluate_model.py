import argparse
import os

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

import os

import torch
from datasets import load_dataset
from sacrebleu import corpus_bleu, corpus_chrf
from tqdm import tqdm
import re
from typing import List
import json

import langcodes


class EvaluateTranslation:

    batch_size = 4

    def __init__(self, model, tok, test_file, prompt, model_name):

        self.model = model
        self.tok = tok
        self.test_file = test_file
        self.prompt = prompt

        self.results = {}
        self.partial_results = {}

        self.model_name = model_name.split("_")[0]
        self.tok.padding_side = "left"
        self.tok.add_bos_token = True

        self.dataset = {"src": [], "tgt": [], "src_lang": None, "tgt_lang": None}
        self.results = {"chrf": 0., "bleu": 0., "blaser": 0.}
        self.partial_results = []

        self.load_data()

    def save_results(self, result_dir):
        test_name = os.path.basename(self.test_file).split(".")[0]
        with open(os.path.join(result_dir, f"res_{self.task}_{test_name}.json"), 'w') as f:
            json.dump(self.results, f, indent=4)

        with open(os.path.join(result_dir, f"partial_res_{self.task}_{test_name}.json"), 'w') as f:
            json.dump(self.partial_results, f, indent=4)

    def load_data(self):
        src_sentences = []
        tgt_sentences = []

        language_pair = self.test_file.split("_")[1]
        src_lang, tgt_lang = language_pair.split("-")
        try:
            if len(self.test_file.split("_")) > 1:
                dataset = load_dataset(self.test_file.split("_")[0],
                                            language_pair,
                                            split='test').to_iterable_dataset()
            else:
                dataset = load_dataset(self.test_file, split='test').to_iterable_dataset()

        except FileNotFoundError:
            raise NotImplementedError(f"The file {self.test_file} was not found in HF."
                                      f"Local test sets are not yet supported.")

        src_sentences = [te[language_pair][src_lang] for te in dataset]
        tgt_sentences = [te[language_pair][tgt_lang] for te in dataset]

        self.dataset = {"src": src_sentences, "tgt": tgt_sentences, "src_lang": src_lang, "tgt_lang": tgt_lang}

    @staticmethod
    def compute_chrf(translated_sentences: List[str], tgt_sentences: List[str]):
        return corpus_chrf(translated_sentences, [tgt_sentences]).score

    @staticmethod
    def compute_bleu(translated_sentences: List[str], tgt_sentences: List[str]):
        return corpus_bleu(translated_sentences, [tgt_sentences]).score

    def translate_sentences(self, src_sentences, src_lang, tgt_lang):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        translated_sentences = []

        prompt_template = self.prompt.replace("[source language]", langcodes.Language(src_lang).language_name())
        prompt_template = prompt_template.replace("[target language]", langcodes.Language(tgt_lang).language_name())
        prompts = [prompt_template.replace("[source sentence]", sentence) for sentence in src_sentences]

        for prompt in tqdm(prompts, desc=f"Translating {src_lang} to {tgt_lang}"):

            inputs = self.tok(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256).input_ids.to(device)
            with torch.no_grad():
                generated = self.model.generate(input_ids=inputs, num_beams=5, max_new_tokens=256)
            translated = self.tok.batch_decode(generated, skip_special_tokens=True)[0].replace(prompt, "").strip()

            translated_sentences.append(translated)
            del inputs, generated, prompt
            torch.cuda.empty_cache()

        return translated_sentences

    def evaluate(self):

        translated_sentences = self.translate_sentences(self.dataset["src"], self.dataset["src_lang"], self.dataset["tgt_lang"])

        if self.dataset["tgt"]:
            self.results["chrf"] = self.compute_chrf(translated_sentences, self.dataset["tgt"])
            self.results["bleu"] = self.compute_bleu(translated_sentences, self.dataset["tgt"])
            self.partial_results = [{"src": src, "tgt": tgt, "pred": pred} for src, tgt, pred in zip(self.dataset["src"], self.dataset["tgt"], translated_sentences)]

        else:
            self.partial_results = [{"src": src, "pred": pred} for src, pred in zip(self.dataset["src"], translated_sentences)]


def get_model_tokenizer(model_name):

    model_path = model_name
    tokenizer_path = model_name
    model_name = model_name.split("/")[-1].replace('-','_')


    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                 low_cpu_mem_usage=True, device_map='auto', offload_folder="offload")

    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        model = model.eval().cuda()
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = model.eval()

    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, return_token_type_ids=False, add_bos_token=False)
    # set llama special tokens
    tok.bos_token = "<s>"
    tok.eos_token = "</s>"
    tok.pad_token = "</s>"
    tok.unk_token = "<unk>"
    tok.padding_side = "right"

    return model_name, model, tok

def run_evaluation_on_task(model, tokenizer, model_name, test_file, prompt, output_dir):

    evaluator = EvaluateTranslation(model, tokenizer, test_file, prompt, model_name)

    evaluator.evaluate()
    evaluator.save_results(output_dir)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama")
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--test_task", type=str, default="gen")
    parser.add_argument("--prompt_file", type=str)
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()

    model_name, model, tok = get_model_tokenizer(args.model_name)

    prompt_file = args.prompt_file
    with open(args.prompt_file, 'r') as f:
        prompts = json.load(f)

    for ex in prompts:
        prompt_id = ex["id"]
        prompt = ex["prompt"]
        print(f"Evaluating original model {model_name}, {prompt_id}")
        output_dir = os.path.join(model_name, prompt_id)
        os.makedirs(output_dir, exist_ok=True)
        run_evaluation_on_task(model, tok, model_name, args.test_file, prompt, output_dir)

