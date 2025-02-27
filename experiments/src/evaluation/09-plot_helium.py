import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
import glob
import collections
import grammar_v_mtllm.utils_fig

# python3 ./scripts/04-plot_basic.py computed/evals/seth-v1-tower---de-en

args = argparse.ArgumentParser()
args.add_argument("glob")
args = args.parse_args()

# load all data from args.dir
data_all = [
    [json.loads(x) for x in open(f)]
    for f in glob.glob(args.glob)
]

KEY_Y = "chrf"

data_all_joined = collections.defaultdict(list)
for data in data_all:
    for x in data:
        data_all_joined[x["prompt_noiser"]].append(x)
data_all = list(data_all_joined.values())

fig, axs = plt.subplots(1, 3, figsize=(9, 2.5))

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
                "prompt_chrf": data[0]["eval_prompt"]["chrf"],
                "prompt_ip": data[0]["eval_prompt"]["ip"],
                "prompt_p": data[0]["eval_prompt"]["p"],
            }
            for data in data_all_local
        ]
        data_local.sort(key=lambda x: x[KEY_X])

        # linear line
        # x = np.linspace(
        #     min([x[KEY_X] for x in data_local]),
        #     max([x[KEY_X] for x in data_local]),
        #     10
        # )
        # y = np.poly1d(np.polyfit([x[KEY_X] for x in data_local], [x[KEY_Y] for x in data_local], 1))
        # ax.plot(x, y(x))
        ax.scatter(
            [x[KEY_X] for x in data_local],
            [x[KEY_Y] for x in data_local],
            marker=".",
            s=70,
            label=langs
            #   + f" ({y[1]:.2f})",
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
python3 experiments/src/evaluation/09-plot_helium.py 'data/evaluated/*/three/test/*-orthographic_*.jsonl'
"""