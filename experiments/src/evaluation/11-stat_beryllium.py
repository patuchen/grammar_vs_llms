import json
import argparse
import numpy as np
import collections
import sys

args = argparse.ArgumentParser()
args.add_argument("data", nargs="+")
args = args.parse_args()

# load all data from args.dir
data_all = [
    [json.loads(x) for x in open(f, "r")]
    for f in args.data
]


def get_bucket_id(x):
    if not x["bucket_id"]: 
        # This is only for `llm` noise. We don't have buckets for it, so we treat 
        # every noised version of a prompt as a separate bucket.
        return x["prompt"]
    # We want to extract bucket id from something like bucket_id-1_prompt_id-mt-03-no_errors_scenario...
    return x["bucket_id"].split("-")[1].split("_")[0]


data_all_joined = collections.defaultdict(list)
for data in data_all:
    for x in data:
        data_all_joined[(get_bucket_id(x), x["prompt_src"])].append(x)
data_all = list(data_all_joined.values())

prompt_to_id = {}
def get_prompt_id(x):
    if x not in prompt_to_id:
        prompt_to_id[x] = f"prompt {len(prompt_to_id)+1}"
    return prompt_to_id[x]


# each file is an individual bucket = one point
data_local = [
    {
        "comet": np.average([x["eval"]["comet"] for x in data]),
        "chrf": np.average([x["eval"]["chrf"] for x in data]),
        "prompt_chrf": np.average([line["eval_prompt"]["chrf"] for line in data]),
        "prompt_ip": np.average([line["eval_prompt"]["ip"] for line in data]),
        "prompt": get_prompt_id(data[0]["prompt_src"]),
    }
    for data in data_all
]

prompts = {x["prompt"] for x in data_local}

for KEY_Y in ["comet", "chrf"]:
    for KEY_X in ["prompt_chrf", "prompt_ip"]:
        stats_a_in_prompts = []
        for prompt_i, prompt in enumerate(sorted(list(prompts))):
            data_local_prompt = [x for x in data_local if x["prompt"] == prompt]
            a = np.corrcoef([x[KEY_X] for x in data_local_prompt], [x[KEY_Y] for x in data_local_prompt])[0, 1]
            stats_a_in_prompts.append(a)
            print(json.dumps({
                "prompt": f"prompt{prompt_i}",
                "key_x": KEY_X,
                "key_y": KEY_Y,
                "slope": a,
                "data_x": [x[KEY_X] for x in data_local_prompt],
                "data_y": [x[KEY_Y] for x in data_local_prompt],
            }), file=sys.stderr)


        print(f"{KEY_X:>15} slope -{KEY_Y:>7} --- {np.average(stats_a_in_prompts):.2f}")

"""
rm -f figures/11-stat_beryllium.{out,err}
for key in "noising_typos_synthetic" "noising_orthographic" "noising_register" "noising_llm" "noising_lexicalphrasal" "noising_LazyUser" "noising_L2"; do
    echo $key;
    echo $key >> figures/11-stat_beryllium.out;
    echo $key >> figures/11-stat_beryllium.err;
    ls data/evaluated/*/three/test/${key}_*.jsonl | wc -l;
    python3 experiments/src/evaluation/11-stat_beryllium.py data/evaluated/*/three/test/${key}_*.jsonl 1>> figures/11-stat_beryllium.out 2>> figures/11-stat_beryllium.err;
done;
"""

