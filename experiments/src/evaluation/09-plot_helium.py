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


KEY_Y = "comet"

data_all_joined = collections.defaultdict(list)
for data in data_all:
    for x in data:
        data_all_joined[x["prompt_noiser"]].append(x)
data_all = list(data_all_joined.values())

fig, axs = plt.subplots(1, 3, figsize=(9, 2.5), sharey=True)

for KEY_X, ax in zip(["prompt_p", "prompt_ip", "prompt_chrf"], axs):
    for langs_i, langs in enumerate(sorted({x["langs"] for data in data_all for x in data})):
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
            linewidth=0,
            alpha=0.5,
            s=20,
            color=grammar_v_mtllm.utils_fig.LANG_TO_COLOR[langs],
        )

        # plot linear fit
        x = np.linspace(
            min([x[KEY_X] for x in data_local]),
            max([x[KEY_X] for x in data_local]),
            10
        )
        y = np.poly1d(np.polyfit(
            [x[KEY_X] for x in data_local],
            [x[KEY_Y] for x in data_local],
            deg=1
        ))
        ax.plot(
            x,
            y(x),
            zorder=10,
            color=grammar_v_mtllm.utils_fig.LANG_TO_COLOR[langs],
            label=grammar_v_mtllm.utils_fig.LANG_TO_NAME[langs] if ax == axs[1] else None,
        )

        if ax == axs[0]:
            ax.set_ylabel("Translation quality")
        elif ax == axs[1]:
            ax.legend(
                frameon=False,            
                handletextpad=0.2,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.2),
                ncol=3,
            )

        ax.set_xlabel({
            "prompt_p": "Perturbation probability",
            "prompt_chrf": "Prompt similarity (surface)",
            "prompt_ip": "Prompt similarity (semantic)",
        }[KEY_X])
        grammar_v_mtllm.utils_fig.turn_off_spines(ax=ax)

        # plot language for the last point

plt.tight_layout(
    rect=[0, 0, 1, 1.05]
)
plt.subplots_adjust(
    wspace=0.05,
)
plt.savefig("figures/09-helium.pdf")
plt.show()

"""
python3 experiments/src/evaluation/09-plot_helium.py data/evaluated/*/three/test/*-orthographic_*.jsonl
"""