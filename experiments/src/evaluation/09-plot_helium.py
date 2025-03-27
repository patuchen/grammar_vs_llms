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

KEY_Y = "chrf"

data_all_joined = collections.defaultdict(list)
for data in data_all:
    for x in data:
        data_all_joined[x["prompt_noiser"]].append(x)
data_all = list(data_all_joined.values())

fig, axs = plt.subplots(1, 3, figsize=(9, 2.5), sharey=True)

for KEY_X, ax in zip(["prompt_p", "prompt_ip", "prompt_chrf"], axs):
    for langs in sorted({x["langs"] for data in data_all for x in data}):
        lang1, lang2 = langs.split("-")
        data_all_local = [
            [x for x in data if x["langs"] == langs]
            for data in data_all
        ]

        data_local = [
            {
                "comet": np.average([x["eval"]["comet"] for x in data]),
                "chrf": np.average([x["eval"]["chrf"] for x in data]),
                "langs": np.average([x["eval"]["langs"][0][0][:2] == lang2 for x in data]),
                "prompt_chrf": np.average([line["eval_prompt"]["chrf"] for line in data]),
                "prompt_ip": np.average([line["eval_prompt"]["ip"] for line in data]),
                "prompt_p": np.average([line["prompt_p"]["orthographic"] for line in data]),
            }
            for data in data_all_local
        ]
        data_local.sort(key=lambda x: x[KEY_X])

        ax.scatter(
            [x[KEY_X] for x in data_local],
            [x[KEY_Y] for x in data_local],
            marker=".",
            s=70,
        )

        ax.text(
            x=[x[KEY_X] for x in data_local][-1],
            y=[x[KEY_Y] for x in data_local][-1],
            s=langs,
            fontsize=8,
            ha="right",
            va="bottom",
        )

        if ax == axs[0]:
            ax.set_ylabel("Translation quality")
        ax.set_xlabel({
            "prompt_p": "Perturbation probability",
            "prompt_chrf": "Prompt distance (surface)",
            "prompt_ip": "Prompt distance (semantic)",
        }[KEY_X])
        grammar_v_mtllm.utils_fig.turn_off_spines(ax=ax)

        # plot language for the last point

plt.tight_layout()
plt.savefig("figures/09-helium.pdf")
plt.show()

"""
python3 experiments/src/evaluation/09-plot_helium.py data/evaluated/*/three/test/*-orthographic_*.jsonl
"""