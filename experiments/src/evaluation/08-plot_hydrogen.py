import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
import collections
import grammar_v_mtllm.utils_fig

args = argparse.ArgumentParser()
args.add_argument("data", nargs="+")
args = args.parse_args()

# load all data from args.dir
data_all = [
    [json.loads(x) for x in open(f)]
    for f in args.data
]

KEY_X = "prompt_ip"
KEY_Y = "comet"

data_all_joined = collections.defaultdict(list)
for data in data_all:
    for x in data:
        data_all_joined[(x["prompt_noiser"], x["prompt_src"])].append(x)
data_all = list(data_all_joined.values())

prompt_to_id = {}
def get_prompt_id(x):
    if x not in prompt_to_id:
        prompt_to_id[x] = f"prompt {len(prompt_to_id)+1}"
    return prompt_to_id[x]


# each file is an individual bucket = one point
data_local = [
    {
        "comet": np.average([x["eval"]["comet"] for x in data]),
        "chrf": np.average([x["eval"]["chrf"] for x in data]),
        # "langs": np.average([x["eval"]["langs"][0][0][:2] == lang2 for x in data]),
        "prompt_chrf": np.average([line["eval_prompt"]["chrf"] for line in data]),
        "prompt_ip": np.average([line["eval_prompt"]["ip"] for line in data]),
        "prompt_p": np.average([line["prompt_p"]["orthographic"] for line in data]),
        # ERROR
        # TODO: this is not true because each file can contain multiple prompt_src?
        "prompt": get_prompt_id(data[0]["prompt_src"]),
    }
    for data in data_all
]

plt.figure(figsize=(3.5, 2.5))
ax = plt.gca()

prompts = {x["prompt"] for x in data_local}

stats_var_in_prompts = []
stats_avg_in_prompts = []

for prompt_i, prompt in enumerate(sorted(list(prompts))):
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
    ax.scatter(
        [x[KEY_X] for x in data_local_prompt],
        [x[KEY_Y] for x in data_local_prompt],
        marker=".",
        s=50,
        color=grammar_v_mtllm.utils_fig.COLORS[prompt_i],
        label=prompt
    )
    ax.plot(
        x,
        y(x),
        color="black",
        zorder=10,
        alpha=0.5,
    )
    stats_var_in_prompts.append(np.var([x[KEY_Y] for x in data_local_prompt]))
    stats_avg_in_prompts.append(np.average([x[KEY_Y] for x in data_local_prompt]))

ax.set_ylabel("Translation quality")
ax.set_xlabel("Similarity to original prompt")

grammar_v_mtllm.utils_fig.turn_off_spines(ax=ax)

plt.legend(
    frameon=False,
    handletextpad=0.2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.4),
    ncol=2,
)
plt.tight_layout(
    rect=[0, 0, 1, 1.1]
)
plt.savefig("figures/08-hydrogen.pdf")
plt.show()

# statistics
print("avg variance within each prompt", np.average(stats_var_in_prompts))
print("variance across averages in each prompt", np.var(stats_avg_in_prompts))

"""
python3 experiments/src/evaluation/08-plot_hydrogen.py data/evaluated/*/three/test/*-orthographic_*.jsonl
"""