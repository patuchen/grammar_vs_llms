import os
from typing import List, Dict, Any
from openai import OpenAI
from anthropic import Anthropic, AnthropicVertex
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from tqdm import tqdm
import httpx

class Model:
    def __init__(self, model: str, gpus: int, mem_percent: float, sampling_params: SamplingParams, system_prompt: str):
        self.model = model
        self.gpus = gpus
        self.sampling_params = sampling_params
        self.system_prompt = system_prompt

        self.short = model.split("/")[1].split("-")[0] if "/" in model else model.split("-")[0]
        self.type = "vllm" if "/" in model else "openai"

        self.llm: LLM | Anthropic | OpenAI = None
        if self.type == "vllm":
            self.llm = LLM(model=model, tensor_parallel_size=gpus, gpu_memory_utilization=mem_percent)

    def __call__(self, prompts: List[str]) -> List[str]:
        return self.generate(prompts)

    def generate(self, model_inputs: list[str], *, quiet: bool = False) -> list[str]:
        raise NotImplementedError()

class EuroLLMModel(Model):
    def generate(self, prompts: list[str], *, quiet: bool = False) -> list[str]:
        system = {
            "role": "system",
            "content": self.system_prompt
        }
        conversations = [
            [system, {"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        responses = self.llm.chat(messages=conversations, sampling_params=self.sampling_params)        
        return [response.outputs[0].text.strip().replace('\n', ' ').replace('\t', ' ') for response in responses]

class TowerModel(Model):
    def generate(self, prompts: list[str], *, quiet: bool = False) -> list[str]:        
        # In the documentation from HF they don't use the system prompt
        conversations = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        responses = self.llm.chat(messages=conversations, sampling_params=self.sampling_params)        
        return [response.outputs[0].text.strip().replace('\n', ' ').replace('\t', ' ') for response in responses]

class AnthropicModel(Model):
    def __init__(self, model: str, gpus: int, sampling_params: SamplingParams, system_prompt: str) -> None:
        """
        This model requires gcloud:
        
        ```
        curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
        tar -xf google-cloud-cli-linux-x86_64.tar.gz
        ./google-cloud-sdk/install.sh
        ./google-cloud-sdk/bin/gcloud init
        ```

        Before using this model run: gcloud auth application-default login

        This will create the credential file here:
        - Linux: $HOME/.config/gcloud/application_default_credentials.json
        - Windows: %APPDATA%\\gcloud\\application_default_credentials.json
        """
        super().__init__(model, gpus, 1, sampling_params, system_prompt)
        load_dotenv()
        self.llm = AnthropicVertex(
            region=os.environ["LOCATION"], 
            project_id=os.environ["PROJECT_ID"],
            max_retries=10,
            timeout=httpx.Timeout(300, connect=300, read=300, write=300),
        )

    def generate(self, prompts: list[str], *, quiet: bool = False) -> list[str]:
        # The system prompt is added by .messages.create
        conversations = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
            for prompt in prompts
        ]
        # TODO add top_p and top_k if used
        responses = [
            self.llm.messages.create(
                model=self.model,
                max_tokens=self.sampling_params.max_tokens,
                temperature=self.sampling_params.temperature,
                system=self.system_prompt,
                messages=conversation,
                stop_sequences=self.sampling_params.stop,                
            ) 
            for conversation in tqdm(conversations, disable=quiet)
        ]
        # TODO check if this is correct
        responses = [response.content[0].text.replace('\n', ' ').replace('\t', ' ') for response in responses]
        return responses
    

class OpenAIModel(Model):
    def __init__(self, model: str, gpus: int, sampling_params: SamplingParams, system_prompt: str) -> None:
        super().__init__(model, gpus, 1, sampling_params, system_prompt)
        load_dotenv()
        self.llm = OpenAI()

    def generate(self, prompts: list[str], *, quiet: bool = False) -> list[str]:
        # Gpt-4o: role "developer" is converted to "system"
        system = {"role": "system", "content": self.system_prompt}
        
        conversations = [
           [system, {"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        # TODO add top_p if used, top_k not supported
        responses = [
            self.llm.chat.completions.create(
                model=self.model,
                messages=conversation,      
                n=self.sampling_params.n,
                max_tokens=self.sampling_params.max_tokens,
                temperature=self.sampling_params.temperature,
                seed=self.sampling_params.seed,
                stop=self.sampling_params.stop
            ) 
            for conversation in tqdm(conversations, disable=quiet)
        ]

        responses = [response.choices[0].message.content.replace('\n', ' ').replace('\t', ' ') for response in responses]
        return responses

def load_model(model: str, gpus: int, mem_percent: float, sampling_params: SamplingParams = None, system_prompt: str = "You are a helpful machine translation assistant.") -> Model:    
    if sampling_params is None:
        sampling_params = default_sampling_params()

    if "gpt" in model.lower():
        return OpenAIModel(model, gpus, sampling_params, system_prompt)
    elif "tower" in model.lower():
        return TowerModel(model, gpus, mem_percent, sampling_params, system_prompt)
    elif "euro" in model.lower():
        return EuroLLMModel(model, gpus, mem_percent, sampling_params, system_prompt)
    elif "claude" in model.lower():
        return AnthropicModel(model, gpus, sampling_params, system_prompt)
    else:
        raise ValueError(f"Model {model} not supported")
    
def default_sampling_params() -> SamplingParams:
    # TODO define reasonable values
    return SamplingParams(
        n=1,
        max_tokens=512,
        temperature=0.05,
        # stop at end of text (can trim after newlines later?)
        stop=["<|im_end|>", "<|eot_id|>"], # TODO make sure this works for all the models
        # top_p=0.9,
        # top_k=0,
        seed=0
    )
