# %%
import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
import collections
import grammar_v_mtllm.utils_fig
import grammar_v_mtllm.utils
import pickle

args = argparse.ArgumentParser()
args.add_argument("data", nargs="+")
args = args.parse_args()

data_all = grammar_v_mtllm.utils.cache_guard("hydrogen", args.data)

KEY_X = "prompt_ip"
KEY_Y = "comet"

def get_bucket_id(bucket_id_value):
    if not bucket_id_value:
        return 0
    # We want to extrack bucket id from something like bucket_id-1_prompt_id-mt-03-no_errors_scenario...
    return bucket_id_value.split("-")[1].split("_")[0]

data_all_joined = collections.defaultdict(list)
for data in data_all:
    for x in data:
        data_all_joined[(get_bucket_id(x["bucket_id"]), x["prompt_src"])].append(x)
data_all = list(data_all_joined.values())

prompt_to_id = {}
def get_prompt_id(x):
    if x not in prompt_to_id:
        if x == "{source_lang}: {source_text}\\n{target_lang}:":
            prompt_to_id[x] = f"prompt minimal"
        else:
            prompt_to_id[x] = f"prompt {len(prompt_to_id)+1}"
    return prompt_to_id[x]


# each file is an individual bucket = one point
data_local = [
    {
        "comet": np.average([x["eval"]["comet"] for x in data]),
        "chrf": np.average([x["eval"]["chrf"] for x in data]),
        "prompt_chrf": np.average([line["eval_prompt"]["chrf"] for line in data]),
        "prompt_ip": np.average([line["eval_prompt"]["ip"] for line in data]),
        # ERROR
        # NOTE: this seems to not be true because each file can contain multiple prompt_src?
        # NOTE: this is fine because we are no long at the file level but collated
        "prompt": get_prompt_id(data[0]["prompt_src"]),
    }
    for data in data_all
]

# %%
plt.figure(figsize=(3.5, 2.5))
ax = plt.gca()

prompts = {x["prompt"] for x in data_local}

stats_var_in_prompts = []
stats_avg_in_prompts = []

for prompt_i, prompt in enumerate(sorted(list(prompts))):
    # disregard this outlier
    data_local_prompt = [x for x in data_local if x["prompt"] == prompt]

    x = np.linspace(
        min([x[KEY_X] for x in data_local_prompt]),
        max([x[KEY_X] for x in data_local_prompt]),
        10
    )
    y = np.poly1d(np.polyfit(
        [x[KEY_X] for x in data_local_prompt],
        [x[KEY_Y] for x in data_local_prompt],
        deg=1
    ))

    if prompt == "prompt minimal":
        ax.axhline(
            y=np.average([x[KEY_Y] for x in data_local_prompt]),
            color="gray",
            linestyle="--",
        )
        ax.text(
            x=0.01,
            y=np.average([x[KEY_Y] for x in data_local_prompt])-0.14,
            s="Minimal\n prompt",
            fontsize=8,
            color="#222",
            ha="left",
            va="bottom",
            # in ax coordinates
            transform=ax.transAxes,
        )
    else:
        ax.scatter(
            [x[KEY_X] for x in data_local_prompt],
            [x[KEY_Y] for x in data_local_prompt],
            marker=".",
            linewidth=0,
            s=40,
            alpha=0.5,
            color=grammar_v_mtllm.utils_fig.COLORS[prompt_i],
        )
        ax.plot(
            x,
            y(x),
            zorder=10,
            color=grammar_v_mtllm.utils_fig.COLORS[prompt_i],
            label=prompt.capitalize()
        )
        stats_var_in_prompts.append(np.var([x[KEY_Y] for x in data_local_prompt]))
        stats_avg_in_prompts.append(np.average([x[KEY_Y] for x in data_local_prompt]))

ax.set_ylabel("Quality (COMET)")
ax.set_xlabel("Similarity to original prompt")

grammar_v_mtllm.utils_fig.turn_off_spines(ax=ax)

plt.legend(
    frameon=False,
    handletextpad=0.1,
    handlelength=0.7,

    loc="upper center",
    bbox_to_anchor=(0.4, 1.4),
    ncol=4,
    columnspacing=0.5,
)
plt.tight_layout(
    rect=[0, 0, 1, 1.1]
)
plt.savefig("figures/08-hydrogen.pdf")
plt.show()

# statistics
print("avg variance within each prompt", np.average(stats_var_in_prompts)*100)
print("variance across averages in each prompt", np.var(stats_avg_in_prompts)*100)

"""
# use all
python3 experiments/src/evaluation/08-plot_hydrogen.py data/evaluated/*/three/test/*.jsonl

noising_typos_synthetic
noising_orthographic
noising_register
noising_llm
noising_lexicalphrasal
noising_LazyUser
noising_L2
"""