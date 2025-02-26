import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pathlib

# python3 ./scripts/04-plot_basic.py computed/evals/seth-v1-tower---de-en

args = argparse.ArgumentParser()
args.add_argument("dir")
args = args.parse_args()


# load all data from args.dir
data_all = [
    [json.loads(x) for x in open(f)]
    for f in pathlib.Path(args.dir).glob("*.jsonl")
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
            "langs": np.average([x["eval"]["langs"] for x in data]),
            "prompt_chrf": data[0]["eval_prompt"]["chrf"],
            "prompt_ip": data[0]["eval_prompt"]["ip"],
            "prompt_p": data[0]["eval_prompt"]["p"],
        }
        for data in data_all_local
    ]

    plt.plot(
        [x["chrf"] for x in data_local],
        [x["prompt_chrf"] for x in data_local],
        marker=".",
        markersize=20,
    )
    for line in data_local:
        plt.text(
            line["chrf"],
            line["prompt_chrf"],
            f"{np.average([x[0][0] == lang2 for x in line[2]]):.0%}\n",
            ha="center", va="center"
        )
    plt.ylabel("Translation quality")
    plt.xlabel("Noise level")
    plt.title((
        args.dir.split("/")[-1].removesuffix(".jsonl")
    ))
        
    plt.text(
        0.05, 0.05, f'% of output in {lang2}',
        transform = plt.gca().transAxes
    )

    plt.tight_layout()

    plt.show()
