from vllm import LLM, SamplingParams

def load_model(model, gpus):

    model = model
    model_short = model.split("/")[1].split("-")[0]

    llm = LLM(model=model, tensor_parallel_size=gpus)

    return llm, model_short


sampling_params = SamplingParams(
    use_beam_search=False,
    best_of=1,
    # Sstop at newlines. This avoids model continuing to generating translations
    stop=["<|im_end|>", "\n"], 
    # epsilon sampling: min_p=0.02
    # min_p=0.02,

    # nucleus sampling: top_p=0.9
    # top_p=0.9,

    # greedy: temp=0
    # temperature=0.05,
    max_tokens=512,
)