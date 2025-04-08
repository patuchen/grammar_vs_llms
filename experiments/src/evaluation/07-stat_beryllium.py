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

def get_bucket_id(bucket_id_value):
    if not bucket_id_value:
        return 0
    # We want to extrack bucket id from something like bucket_id-1_prompt_id-mt-03-no_errors_scenario...
    return bucket_id_value.split("-")[1].split("_")[0]


data_all_joined = collections.defaultdict(list)
for data in data_all:
    for x in data:
        data_all_joined[(get_bucket_id(x["bucket_id"]), x["prompt_src"])].append(x)
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
        # "langs": np.average([x["eval"]["langs"][0][0][:2] == lang2 for x in data]),
        "prompt_chrf": np.average([line["eval_prompt"]["chrf"] for line in data]),
        "prompt_ip": np.average([line["eval_prompt"]["ip"] for line in data]),
        # "prompt_p": np.average([line["prompt_p"]["orthographic"] for line in data]),
        # ERROR
        # TODO: this is not true because each file can contain multiple prompt_src?
        "prompt": get_prompt_id(data[0]["prompt_src"]),
    }
    for data in data_all
]

prompts = {x["prompt"] for x in data_local}

for KEY_Y in ["comet", "chrf"]:
    for KEY_X in ["prompt_chrf", "prompt_ip"]:
        stats_a_in_prompts = []
        for prompt_i, prompt in enumerate(sorted(list(prompts))):
            # disregard this outlier
            if prompt_i >= 4:
                continue
            data_local_prompt = [x for x in data_local if x["prompt"] == prompt]
            # print(f"Number of data points for {KEY_X} - {KEY_Y}: {len(data_local_prompt)}")
            # coefficient
            # a = np.polyfit(
            #     [x[KEY_X] for x in data_local_prompt],
            #     [x[KEY_Y] for x in data_local_prompt],
            #     deg=1
            # )[0]
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

        # if KEY_Y == "comet":
        #     stats_a_in_prompts = np.array(stats_a_in_prompts) * 100

        print(f"{KEY_X:>15} slope -{KEY_Y:>7} --- {np.average(stats_a_in_prompts):.2f}")
        # if KEY_X == "prompt_chrf":
        # if KEY_X == "prompt_ip":
        #     print(f"{KEY_X:>15} slope -{KEY_Y:>7} --- {np.average(stats_a_in_prompts):.1f}")

"""
rm -f figures/07-stat_beryllium.{out,err}
for key in "noising_typos_synthetic" "noising_orthographic" "noising_register" "noising_llm" "noising_lexicalphrasal" "noising_LazyUser" "noising_L2"; do
    echo $key;
    echo $key >> figures/07-stat_beryllium.out;
    echo $key >> figures/07-stat_beryllium.err;
    ls data/evaluated/*/three/test/${key}_*.jsonl | wc -l;
    python3 experiments/src/evaluation/07-stat_beryllium.py data/evaluated/*/three/test/${key}_*.jsonl 1>> figures/07-stat_beryllium.out 2>> figures/07-stat_beryllium.err;
done;
"""

