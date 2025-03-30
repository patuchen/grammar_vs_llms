from noising_functions import ComposeNoise
from typing import Dict
import numpy as np
# from evaluate import load
import json 
from utils import define_noise_profiles, get_noise_profile_key, scenarios
from sampling_prompts_by_similarity import bucket_prompts_by_chrf

np.random.seed(42)
# perplexity = load("perplexity", module_type="metric")

def format_prompt_entry(prompt, prompt_metadata, noised_prompt, noised_prompt_id, score, scenario_key, noise_profile):
    '''
    Format the output of the noised prompt.
    '''
    noise_type_probabilities = {noise_type: noise_profile[noise_type]["p"] for noise_type in noise_profile}

    prompt_id = prompt_metadata["prompt_id"]
    formatted = {
        "noised_prompt_id": f"prompt-{prompt_id}_scenario-{scenario_key}_noised-{noised_prompt_id}",
        "noised_prompt": noised_prompt,
        "perplexity": score,
        "noise_type_parameters": noise_type_probabilities,
        "prompt_noiser": scenario_key
    }
    # Check that prompt metadata doesn't contain any keys that are already in the formatted output
    assert not any(k in formatted for k in prompt_metadata)
    formatted.update(prompt_metadata)

    return formatted


def noise_and_score_prompts(prompts: list, noise_profile: Dict[str, Dict], scenario_key: str, n_samples: int, score_perplexity: bool = False) -> Dict[str, Dict[str, float]]:
    '''
    Generate and score noised prompts.
    prompts: [{"id": <prompt_id>, "prompt": <prompt>, "metadata1": <metadata1>, "metadata2": <metadata2>, ...}]
    '''
    noiser_obj = ComposeNoise(noise_profile)
    noised_prompt_output = []

    for prompt_data in prompts:
        prompt = prompt_data["prompt"]
        prompt_metadata = {k: v for k, v in prompt_data.items() if k not in ["prompt"]}
        noised_prompts = [noiser_obj.noise(prompt) for _ in range(n_samples)]
        noised_scores = perplexity.compute(predictions=noised_prompts, model_id = "gpt2")["perplexities"] if score_perplexity else [-1]*n_samples
        for noised_prompt_id, (noised_prompt, score) in enumerate(zip(noised_prompts, noised_scores)):
            noised_prompt_output.append(format_prompt_entry(prompt, prompt_metadata, noised_prompt, noised_prompt_id, score, scenario_key, noise_profile))

    return noised_prompt_output


def generate_noised_prompts_for_scenario(scenario_key: str, prompts_file: str, n_samples: int, score_perplexity: bool = False, outfile: str = None) -> Dict[str, Dict[str, float]]:
    '''
    Generate noised prompts.
    '''
    with open(prompts_file, "r") as f:
        prompts = json.load(f)
    noise_profiles = define_noise_profiles(scenario_key)
    all_outputs = []
    for noise_profile in noise_profiles:
        key = get_noise_profile_key(noise_profile)
        # print(key)
        output = noise_and_score_prompts(prompts, noise_profile, key, n_samples, score_perplexity)
        all_outputs.extend(output)

    if outfile:
        with open(outfile, "w") as f:
            json.dump(all_outputs, f, indent=4)
    return output


if __name__ == "__main__":
    ## Toy example
    # prompts = {"1": "This is a test prompt.", "2": "This is another test prompt."}
    # noise_profile = {'orthographic': {'p': 0.5, 'subtype_distribution': {'natural_typos': 0.03, 'insertion': 0.17, 'omission': 0.37, 'transposition': 0.05, 'substitution': 0.38}}}
    # scenario_key = "orthographic"
    # n_samples = 2
    # noised_prompts = noise_and_score_prompts(prompts, noise_profile, scenario_key, n_samples, score_perplexity=True)
    # print(*noised_prompts, sep="\n")

    ## Generating noised versions of mt_base.json with scenario "orthographic" over different values of probability p
    # scenarios = ["typos_synthetic", "orthographic", "lexicalphrasal", "L2"]

    # By default, scenarios contains all 6 scenarios as defined in utils.py
    for scenario in scenarios:
        print(f"Generating noised prompts for scenario {scenario}")
        # prompts_file = "../../prompts/mt_base.json"
        filename = "gemba_base"
        prompts_file = f"../../prompts/{filename}.json"
        n_samples = 20
        outfile = f"../../noised_prompts/{filename}_noised_{scenario}.json"
        generate_noised_prompts_for_scenario(scenario, prompts_file, n_samples, score_perplexity=False, outfile=outfile)

        print(f"Bucketing noised prompts for scenario {scenario}")
        bucketed_outfile = f"../../bucketed_noised_prompts/{filename}_noised_{scenario}_bucketed.json"
        num_buckets = 10
        bucket_prompts_by_chrf(outfile, bucketed_outfile, num_buckets)

