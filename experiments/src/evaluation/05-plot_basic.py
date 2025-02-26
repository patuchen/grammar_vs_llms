import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
import glob

# python3 ./scripts/04-plot_basic.py computed/evals/seth-v1-tower---de-en

args = argparse.ArgumentParser()
args.add_argument("glob")
args = args.parse_args()


# load all data from args.dir
data_all = [
    [json.loads(x) for x in open(f)]
    for f in glob.glob(args.glob)
]

for langs in {x["langs"] for data in data_all for x in data}:
    lang1, lang2 = langs.split("-")
    data_all_local = [
        [x for x in data if x["langs"] == langs]
        for data in data_all
    ]

    data_local = [
        {
            "comet": np.average([x["eval"]["comet"] for x in data]),
            "chrf": np.average([x["eval"]["chrf"] for x in data]),
            "langs": np.average([x["eval"]["langs"][0][0] == lang2 for x in data]),
            # np.average([x["eval"]["langs"] for x in data]),
            "prompt_chrf": data[0]["eval_prompt"]["chrf"],
            "prompt_ip": data[0]["eval_prompt"]["ip"],
            "prompt_p": data[0]["eval_prompt"]["p"],
        }
        for data in data_all_local
    ]

    plt.plot(
        [x["prompt_chrf"] for x in data_local],
        [x["chrf"] for x in data_local],
        marker=".",
        markersize=20,
        label=langs,
    )
    for line in data_local:
        plt.text(
            line["prompt_chrf"],
            line["chrf"],
            f"{line['langs']:.0%}\n",
            ha="center", va="center"
        )
    plt.ylabel("Translation quality")
    plt.xlabel("Noise level")

plt.legend(
    # no frame
    frameon=False,
)
plt.title((args.glob.split("/")[-1].removesuffix(".jsonl")))
plt.tight_layout()
plt.show()


"""
python3 experiments/src/evaluation/05-plot_basic.py data/evaluated/base_micro_test_results.jsonl
"""