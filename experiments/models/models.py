from vllm import LLM, SamplingParams
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

MODELS = {
    "gpt4": "gpt-4o-mini",
    "claude": "claude-3-5-sonnet-20241022",
    "eurollm": "utter-project/EuroLLM-9B-Instruct",
    "tower": "Unbabel/TowerInstruct-13B-v0.1",
}

class OpenModel:
    def __init__(self, model: str, sampling_params: SamplingParams, system_prompt: str = "You are a helpful machine translation assistant.") -> None:
        self._model = LLM(model=model)
        self._sampling_params = sampling_params
        self._system_prompt = system_prompt 

class EuroLLMModel(OpenModel):
    def __call__(self, prompts: list[str] | str):
        system = {
            "role": "system",
            "prompt": self._system_prompt
        }
        conversations = [
            [system, {"role": "user", "prompt": prompt}]
            for prompt in prompts
        ]
        responses = self._model.chat(messages=conversations, sampling_params=self._sampling_params)        
        return [response.outputs[0].text for response in responses]

class TowerModel(OpenModel):
    def __call__(self, prompts: list[str] | str):        
        # In the documentation from HF they don't use the system prompt
        conversations = [
            [{"role": "user", "prompt": prompt}]
            for prompt in prompts
        ]
        responses = self._model.chat(messages=conversations, sampling_params=self._sampling_params)
        return [response.outputs[0].text for response in responses]

class AnthropicModel:
    def __init__(self, model, sampling_params: SamplingParams, system_prompt: str = "You are a helpful machine translation assistant."):
        load_dotenv()
        # self._api_key = os.getenv("ANTHROPIC_API_KEY")
        self._model = Anthropic() # TODO check if the api key is loaded
        self._sampling_params = sampling_params
        self._system_prompt = system_prompt 
        self._model_name = model

    def __call__(self, prompts: list[str]):
        conversations = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
            for prompt in prompts
        ]
        # TODO add top_p and top_k if used
        responses = [
            self._model.messages.create(
                model=self._model_name,
                max_tokens=self._sampling_params.max_tokens,
                temperature=self._sampling_params.temperature,
                system=self._system_prompt,
                messages=conversation
            ) 
            for conversation in conversations
        ]
        responses = [response.content for response in responses]
        return responses

class OpenAIModel:
    def __init__(self, model, sampling_params: SamplingParams, system_prompt: str = "You are a helpful machine translation assistant."):
        load_dotenv()
        # self._api_key = os.getenv("OPENAI_API_KEY")
        self._model = OpenAI() # TODO check if the api key is loaded
        self._sampling_params = sampling_params
        self._system_prompt = system_prompt 
        self._model_name = model

    def __call__(self, prompts: list[str]) -> list[str]:
        # TODO is it system or developer? Doc ambiguous
        system = {"role": "system", "content": self._system_prompt}
        # system = {"role": "developer", "content": self._system_prompt}
        
        conversations = [
           [system, {"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        # TODO add top_p if used, top_k not supported
        responses = [
            self._model.chat.completions.create(
                model=self._model_name,
                messages=conversation,      
                n=self._sampling_params.n,
                max_tokens=self._sampling_params.max_tokens,
                temperature=self._sampling_params.temperature,
                seed=self._sampling_params.seed,
            ) 
            for conversation in conversations
        ]

        responses = [response.choices[0].message for response in responses]
        return responses


def load_model(name: str, params: SamplingParams = None, system_prompt: str = None) -> OpenModel | AnthropicModel | OpenAIModel:
    classes = {
        "gpt4": OpenAIModel,
        "claude": AnthropicModel,
        "eurollm": EuroLLMModel,
        "tower": TowerModel,
    }
    if params is None:
        params = default_sampling_params()
    if system_prompt is None:
        system_prompt = "You are a helpful machine translation assistant."
    return classes[name](model=MODELS[name], sampling_params=params, system_prompt=system_prompt)

def default_sampling_params():
    # TODO define reasonable values
    return SamplingParams(
        n=1,
        max_tokens=512,
        temperature=0,
        # top_p=0.9,
        # top_k=0,
        seed=0
    )
