import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
import scipy.stats


def confidence_interval(data, confidence=0.99):
    return scipy.stats.t.interval(
        confidence=confidence,
        df=len(data)-1,
        loc=np.mean(data),
        scale=scipy.stats.sem(data)
    )

# python3 ./scripts/04-plot_basic.py computed/evals/seth-v1-tower---de-en.jsonl -f mt-01
# python3 ./scripts/04-plot_basic.py computed/evals/kathy-v1---de-en.jsonl -f prompt_01


args = argparse.ArgumentParser()
args.add_argument("jsonl")
args.add_argument("-f", "--filter", default="")
args = args.parse_args()

data = [
    json.loads(x) for x in open(args.jsonl)
]
data = [
    x for x in data
    if args.filter in x["fname"]
]
data = [
    (
        int(x["fname"].removesuffix(".csv").split("_")[-1]),
        np.average(x["score"]),
        x["langs"]
    )
    for x in data
]
data.sort(key=lambda x: x[0])

plt.plot(
    [x[0] for x in data],
    [x[1] for x in data],
    marker=".",
    markersize=20,
)
for line in data:
    plt.text(
        line[0],
        line[1]+2,
        f"{np.average([x[0][0] == "en" for x in line[2]]):.0%}",
        ha="center", va="center"
    )
plt.ylabel("ChrF")
plt.xlabel("Noise level")
plt.title((
    args.jsonl.removeprefix("computed/evals/").removesuffix(".jsonl")
    + " FILTER " + args.filter
))
plt.tight_layout()

plt.show()
