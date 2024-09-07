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

# python3 ./scripts/04-plot_basic.py computed/evals_chrf/gpt-4o-mini---mt-01---cs-uk.jsonl
# python3 ./scripts/04-plot_basic.py computed/evals_chrf/seth-v1-tower---cs-uk.jsonl -f mt-01
# python3 ./scripts/04-plot_basic.py computed/evals_chrf/seth-v1-eurollm---cs-uk.jsonl -f mt-01

# python3 ./scripts/04-plot_basic.py computed/evals_comet/gpt-4o-mini---mt-01---cs-uk.jsonl
# python3 ./scripts/04-plot_basic.py computed/evals_comet/gpt-4o-mini---mt-02---cs-uk.jsonl
# python3 ./scripts/04-plot_basic.py computed/evals_comet/gpt-4o-mini---mt-03---cs-uk.jsonl
# python3 ./scripts/04-plot_basic.py computed/evals_comet/gpt-4o-mini---mt-04---cs-uk.jsonl

# python3 ./scripts/04-plot_basic.py computed/evals_comet/gpt-4o-mini---mt-01---cs-uk.jsonl
# python3 ./scripts/04-plot_basic.py computed/evals_comet/gpt-4o-mini---mt-01---de-en.jsonl
# python3 ./scripts/04-plot_basic.py computed/evals_comet/gpt-4o-mini---mt-01---en-zh.jsonl


args = argparse.ArgumentParser()
args.add_argument("jsonl")
args.add_argument("-f", "--filter", default="")
args = args.parse_args()

TARGET_LANG = (
    "uk" if "cs-uk" in args.jsonl else
    "en" if "de-en" in args.jsonl else
    "zh" if "en-zh" in args.jsonl else
    None
)

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
        line[1],
        f"{np.average([x[0][0] == TARGET_LANG for x in line[2]]):.0%}\n",
        ha="center", va="center"
    )
plt.ylabel("ChrF" if "chrf" in args.jsonl else "COMET-22-DA")
plt.xlabel("Noise level")
plt.title((
    args.jsonl.split("/")[-1].removesuffix(".jsonl")
    + ((" FILTER " + args.filter) if args.filter else "")
))
     
plt.text(
    0.05, 0.05, f'% of output in {TARGET_LANG}',
    # horizontalalignment='left',
    # verticalalignment='center',
    transform = plt.gca().transAxes
)

plt.tight_layout()

plt.show()
