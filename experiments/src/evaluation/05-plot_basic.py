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

KEY_X = "prompt_p"
KEY_Y = "chrf"

data_all_joined = collections.defaultdict(list)
for data in data_all:
    for x in data:
        data_all_joined[x["prompt_noiser"]].append(x)
data_all = list(data_all_joined.values())

for langs in sorted({x["langs"] for data in data_all for x in data}):
    print(langs)
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
    # data_local.sort(key=lambda x: x["prompt_chrf"])



    # linear line
    x = np.linspace(
        min([x[KEY_X] for x in data_local]),
        max([x[KEY_X] for x in data_local]),
        10
    )
    y = np.poly1d(np.polyfit([x[KEY_X] for x in data_local], [x[KEY_Y] for x in data_local], 1))
    plt.plot(x, y(x))
    plt.scatter(
        [x[KEY_X] for x in data_local],
        [x[KEY_Y] for x in data_local],
        marker=".",
        s=70,
        # plot lang and slope
        label=langs + f" ({y[1]:.2f})",
    )


    for line in data_local:
        if line[KEY_X] >= max([x[KEY_X] for x in data_local])*0.99:
            continue
        if line[KEY_X] <= min([x[KEY_X] for x in data_local])*1.01:
            continue
        plt.text(
            line[KEY_X],
            line[KEY_Y],
            f"{line['langs']:.0%}\n",
            ha="center", va="center"
        )
    plt.ylabel("Translation quality")
    plt.xlabel("Noising p")

plt.legend(
    frameon=False,
)
plt.title((args.glob.removeprefix("data/evaluated/").removesuffix(".jsonl")))
plt.tight_layout()
plt.show()