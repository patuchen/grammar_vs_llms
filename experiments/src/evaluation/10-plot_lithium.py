import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
import collections
import grammar_v_mtllm.utils_fig

args = argparse.ArgumentParser()
args.add_argument("key_y", choices=["langs", "comet", "chrf"])
args.add_argument("data", nargs="+")
args = args.parse_args()

# load all data from args.dir
data_all = [
    [json.loads(x) for x in open(f, "r")]
    for f in args.data
]

KEY_X = "prompt_ip"

def get_bucket_id(bucket_id_value):
    if not bucket_id_value:
        return 0
    # We want to extrack bucket id from something like bucket_id-1_prompt_id-mt-03-no_errors_scenario...
    return bucket_id_value.split("-")[1].split("_")[0]

data_all_joined = collections.defaultdict(list)
for data in data_all:
    for x in data:
        data_all_joined[get_bucket_id(x["bucket_id"])].append(x)
data_all = list(data_all_joined.values())

fig, axs = plt.subplots(1, 3, figsize=(9, 2.5), sharey=True)

for langs, ax in zip(sorted({x["langs"] for data in data_all for x in data}), axs):
    ax.text(
        0.95, 0.05,
        grammar_v_mtllm.utils_fig.LANG_TO_NAME[langs],
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
    )

    if ax == axs[0]:
        if args.key_y == "langs":
            ax.set_ylabel("Output in target language")
        elif args.key_y == "comet":
            ax.set_ylabel("Translation Quality (COMET)")
        elif args.key_y == "chrf":
            ax.set_ylabel("Translation Quality (ChrF)")
    ax.set_xlabel({
        "prompt_p": "Perturbation probability",
        "prompt_chrf": "Prompt similarity (surface)",
        "prompt_ip": "Prompt similarity (semantic)",
    }[KEY_X])
    grammar_v_mtllm.utils_fig.turn_off_spines(ax=ax)
    lang1, lang2 = langs.split("-")
    
    print(sorted({x["model"] for data in data_all for x in data}))
    for model_i, model in enumerate(sorted({x["model"] for data in data_all for x in data})):
        data_all_local = [
            [x for x in data if x["langs"] == langs and x["model"] == model]
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
            [x[args.key_y] for x in data_local],
            marker=".",
            s=40,
            alpha=0.5,
            linewidth=0,
        )

        data_xy = list(zip([x[KEY_X] for x in data_local], [x[args.key_y] for x in data_local]))
        data_xy = [(x, y) for x,y in data_xy if not np.isnan(x) and not np.isnan(y) and x >= 0.76]
        data_x, data_y = zip(*data_xy)


        # plot linear fit
        x = np.linspace(
            min(data_x),
            max(data_x),
            10
        )

        y = np.poly1d(np.polyfit(
            data_x,
            data_y,
            deg=1
        ))
        model_name = model.replace("gemini", "Gemini").replace("gpt", "GPT4o")
        ax.plot(
            x,
            y(x),
            zorder=10,
            color=grammar_v_mtllm.utils_fig.COLORS[model_i],
            label=model_name if ax == axs[1] else None,
        )

        if ax == axs[1]:
            ax.legend(
                frameon=False,            
                handletextpad=0.2,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.2),
                ncol=6,
            )

plt.tight_layout(
    rect=[0, 0, 1, 1.05]
)
plt.subplots_adjust(
    wspace=0.05,
)
plt.savefig(f"figures/10-lithium_{args.key_y}.pdf")
plt.show()

"""
python3 experiments/src/evaluation/10-plot_lithium.py langs data/evaluated/*/three/test/*orthographic_*.jsonl
python3 experiments/src/evaluation/10-plot_lithium.py comet data/evaluated/*/three/test/*orthographic_*.jsonl
python3 experiments/src/evaluation/10-plot_lithium.py chrf data/evaluated/*/three/test/*orthographic_*.jsonl
"""