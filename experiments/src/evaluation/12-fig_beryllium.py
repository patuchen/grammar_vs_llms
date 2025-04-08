# %%

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import collections
import grammar_v_mtllm.utils_fig
import pickle

args = argparse.ArgumentParser()
args.add_argument("data", nargs="+")
args = args.parse_args()

# load all data from args.dir
data_all = [
    (f, [json.loads(x) for x in open(f, "r")])
    for f in args.data
]

with open("tmp.pkl", "wb") as f:
    pickle.dump(data_all, f)

with open("../../../tmp.pkl", "rb") as f:
    data_all = pickle.load(f)

def get_bucket_id(bucket_id_value):
    if not bucket_id_value:
        return 0
    # We want to extrack bucket id from something like bucket_id-1_prompt_id-mt-03-no_errors_scenario...
    return bucket_id_value.split("-")[1].split("_")[0]

def get_category(fname):
    if fname is None:
        return None
    for key in ["noising_typos_synthetic", "noising_orthographic", "noising_register", "noising_llm", "noising_lexicalphrasal", "noising_LazyUser", "noising_L2"]:
        if key in fname:
            return key
    
    if "no_errors" in fname:
        return None
    raise Exception("No category found for this file: " + fname)

data_all_joined = collections.defaultdict(lambda: collections.defaultdict(list))
for fname, data in data_all:
    category = get_category(fname)
    for x in data:
        # ignore no_errors category
        if category is None:
            continue
        data_all_joined[category][get_bucket_id(x["bucket_id"])].append(x)

data_all = {
    noiser: list(noiser_v.values())
    for noiser, noiser_v in data_all_joined.items()
}

# %%


# each file is an individual bucket = one point
data_local = {
    noiser: [
        {
            "comet": np.average([x["eval"]["comet"] for x in data]),
            "chrf": np.average([x["eval"]["chrf"] for x in data]),
            "prompt_chrf": np.average([line["eval_prompt"]["chrf"] for line in data]),
            "prompt_ip": np.average([line["eval_prompt"]["ip"] for line in data]),
        }
        for data in noiser_v
    ]
    for noiser, noiser_v in data_all.items()
}

KEY_X = "prompt_ip"
KEY_Y = "comet"

for noiser, noiser_data in data_local.items():
    # disregard prompt information
    data_x = [x[KEY_X] for x in noiser_data]
    data_y = [x[KEY_Y] for x in noiser_data]
    a = np.polyfit(
        data_x,
        data_y,
        deg=1
    )[0]
    plt.scatter(
        data_x,
        data_y,
        label=noiser,
        alpha=0.5,
        s=40,
        linewidth=0,
    )

plt.xlabel(KEY_X)
plt.ylabel(KEY_Y)
plt.legend()
grammar_v_mtllm.utils_fig.turn_off_spines()
plt.tight_layout()
plt.show()

        
"""
python3 experiments/src/evaluation/12-fig_beryllium.py data/evaluated/*/three/test/*.jsonl
"""

