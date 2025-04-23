# %%

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import collections
import grammar_v_mtllm.utils_fig
import pickle

# args = argparse.ArgumentParser()
# args.add_argument("data", nargs="+")
# args = args.parse_args()

# data_all = [
#     (f, [json.loads(x) for x in open(f, "r")])
#     for f in args.data
# ]
# with open("cache_beryllium.pkl", "wb") as f:
#     pickle.dump(data_all, f)

with open("cache_beryllium.pkl", "rb") as f:
    data_all = pickle.load(f)

def get_bucket_id(x):
    if not x["bucket_id"]: 
        # This is only for `llm` noise. We don't have buckets for it, so we treat 
        # every noised version of a prompt as a separate bucket.
        return x["prompt"]
    # We want to extract bucket id from something like bucket_id-1_prompt_id-mt-03-no_errors_scenario...
    return x["bucket_id"].split("-")[1].split("_")[0]

def get_category(fname):
    if fname is None:
        return None
    for key in ["noising_typos_synthetic", "noising_orthographic", "noising_register", "noising_llm", "noising_lexicalphrasal", "noising_LazyUser", "noising_L2"]:
        if key in fname:
            return key
    
    if "no_errors" in fname:
        return None
    raise Exception("No category found for this file: " + fname)

data_all_joined = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
# category: {prompt_id: [x]}
for fname, data in data_all:
    category = get_category(fname)
    for x in data:
        # ignore no_errors category
        if category is None:
            continue
        data_all_joined[category][x["prompt_src"]][get_bucket_id(x)].append(x)


prompts = sorted(data_all_joined["noising_typos_synthetic"].keys())
prompt2id = {prompt: i for i, prompt in enumerate(prompts)}

data_all = {
    noiser: {
        prompt2id[prompt]: list(noiser_prompt.values()) # This is a list of all buckets for this prompt for this noiser
        for prompt, noiser_prompt in noiser_v.items()    
    }
    for noiser, noiser_v in data_all_joined.items()
}

# %%
data_local = {
    noiser: {
        prompt_id: [
            {
                "comet": np.average([x["eval"]["comet"] for x in data]),
                "chrf": np.average([x["eval"]["chrf"] for x in data]),
                "prompt_chrf": np.average([line["eval_prompt"]["chrf"] for line in data]),
                "prompt_ip": np.average([line["eval_prompt"]["ip"] for line in data]),
            }
            for data in prompt_data
        ]
        for prompt_id, prompt_data in noiser_prompt.items()
    } 
    for noiser, noiser_prompt in data_all.items()
}

# %%
KEY_X = "prompt_ip"
KEY_Y = "comet"

# %%
# Let's find correlations between KEY_X and KEY_Y for each noiser for each prompt

x_y_coords = collections.defaultdict(dict) # {noiser: {prompt_id: [(x, y)]}}
correlations = collections.defaultdict(dict) # {noiser: {prompt_id: correlation}}
for noiser, noiser_data in data_local.items():
    for prompt_id, prompt_data in noiser_data.items():
        data_x = [x[KEY_X] for x in prompt_data]
        data_y = [x[KEY_Y] for x in prompt_data]
        if len(data_x) < 2:
            continue
        x_y_coords[noiser][prompt_id] = list(zip(data_x, data_y))
        correlations[noiser][prompt_id] = np.corrcoef(data_x, data_y)[0][1]
        print(f"Noiser: {noiser}, prompt_id: {prompt_id}, correlation: {correlations[noiser][prompt_id]}")

print(f"X_Y coords: {x_y_coords}")
print(f"Correlations: {correlations}")
# correlations = {'noising_L2': {2: 0.1373168287131604, 3: 0.4700333996390263, 0: 0.7096968862085662, 1: 0.4600727344309935}, 'noising_LazyUser': {2: 0.5181376508183871, 3: 0.3639372934737929, 0: 0.9377710564937806, 1: 0.7651680957633882}, 'noising_lexicalphrasal': {2: -0.6722729423598935, 3: 0.29783161248693824, 0: 0.5067378734374537, 1: 0.5859266971752077}, 'noising_llm': {2: 0.5699810847621477, 3: 0.8629553432688191, 0: 0.8370846316192802, 1: 0.7191985872244931}, 'noising_orthographic': {2: 0.6713015246402438, 3: -0.012585437401121542, 0: 0.7053778782097951, 1: 0.5840533253879365}, 'noising_register': {2: -0.16999181952559456, 3: 0.30585922578452396, 0: 0.7444589894701672, 1: 0.8782546386807204}, 'noising_typos_synthetic': {2: 0.6117439958058595, 3: 0.6770739339088172, 0: 0.8640981658055105, 1: 0.9237912556873724}}

# %%

fig, axs = plt.subplots(1, 1, figsize=(8, 3), sharex=True, sharey=True)
grammar_v_mtllm.utils_fig.turn_off_spines(ax=axs)

noisers = sorted(correlations.keys())
prompts = sorted(correlations[noisers[0]].keys())

# Noisers are rows, prompts are columns
heatmap = np.zeros((len(noisers), len(prompts)))
for i, noiser in enumerate(noisers):
    for j, prompt in enumerate(prompts):
        heatmap[i][j] = correlations[noiser][prompt]

cax = axs.matshow(heatmap, cmap="coolwarm", vmin=-1, vmax=1)
axs.set_xticks(range(len(prompts)))
axs.set_yticks(range(len(noisers)))
axs.set_xticklabels(prompts, rotation=45, ha="left")
axs.set_yticklabels(noisers)
axs.set_xlabel("Prompt ID")
axs.set_ylabel("Noiser")

# Add colorbar
cbar = fig.colorbar(cax, ax=axs)
cbar.set_label("Correlation")
plt.tight_layout()
plt.savefig("figures/14_carbon_heatmap_correlations.png", dpi=300)
plt.show()

# %%
        
"""
python3 experiments/src/evaluation/14-fig_carbon.py data/evaluated/*/three/test/*.jsonl
"""

