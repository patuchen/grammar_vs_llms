from vllm import LLM, SamplingParams
from openai import OpenAI
from tqdm import tqdm
from typing import List, Dict, Any
import os

class Model:

    def __init__(self, model, gpus):
        self.model = model
        self.short = model.split("/")[1].split("-")[0] if "/" in model else model.split("-")[0]
        self.gpus = gpus
        self.type = "vllm" if "/" in model else "openai"
        # Initialize the model
        if self.type == "vllm":
            self.llm = LLM(model=self.model, tensor_parallel_size=self.gpus)
        elif self.type == "openai":
            self.llm = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"]
            )

    def generate(self, model_inputs):
        if self.type == "openai":
            # TODO: define system prompt
            messages = [
                [{"role": "system", "content": "You are a helpful machine translation assistant."},
                 {"role": "user", "content": prompt}]
                for prompt in model_inputs
            ]
            translations = []
            for msg in tqdm(messages):
                response = self.llm.chat.completions.create(
                    model=self.model,
                    messages=msg,
                    temperature=0.05,
                    max_tokens=512)

                translation = response.choices[0].message.content.strip()
                translations.append(translation.replace('\n', ' ').replace('\t', ' '))

            return translations

        # vllm
        elif self.type == "vllm":
            outputs = self.llm.generate(model_inputs, sampling_params)
            translations = [o.outputs[0].text.strip().replace('\n', ' ').replace('\t', ' ') for o in outputs]
            return translations


sampling_params = SamplingParams(
    # stop at end of text (can trim after newlines later?)
    stop=["<|im_end|>", "<|eot_id|>"],
    temperature=0.05,
    max_tokens=512,
)