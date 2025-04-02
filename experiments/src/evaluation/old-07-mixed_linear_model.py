import statsmodels.formula.api as smf
import argparse
import json
import collections
import numpy as np
import pandas as pd

args = argparse.ArgumentParser()
args.add_argument("data", nargs="+")
args = args.parse_args()

# load all data from args.dir
data_all = [
    [json.loads(x) for x in open(f)]
    for f in args.data
]

data_all_joined = collections.defaultdict(list)
for data in data_all:
    for x in data:
        data_all_joined[(x["prompt_noiser"], x["model"])].append(x)
data_all = list(data_all_joined.values())

data_pd = []

prompt_to_id = {}
def get_prompt_id(x):
    if x not in prompt_to_id:
        prompt_to_id[x] = f"mx-{len(prompt_to_id)}"
    return prompt_to_id[x]
        

for langs in sorted({x["langs"] for data in data_all for x in data}):
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
            "data": data,
        }
        for data in data_all_local
    ]

    for data_vars in data_local:
        data_pd.append({
            "TranslationQuality": data_vars["chrf"],
            "PromptDistance": data_vars["prompt_chrf"],
            "Language": langs,
            "Model": data_vars["data"][0]["model"],
            "Prompt": get_prompt_id(data_vars["data"][0]["prompt_src"]),
        })

# turn to dataframe
data_pd = pd.DataFrame(data_pd)

# create column group with concatenated Language, Model and Prompt
data_pd["Group"] = data_pd["Language"] + "|" + data_pd["Model"] + "|" + data_pd["Prompt"]

model = smf.mixedlm(
    "TranslationQuality ~ PromptDistance + Model + C(Prompt)",
    data_pd,
    groups=data_pd["Language"],
    re_formula="1"
)
# model = smf.ols(
#     "TranslationQuality ~ PromptDistance + Language + Model",
#     data_pd,
# )
print(model.fit().summary())

"""
python3 experiments/src/evaluation/07-mixed_linear_model.py 'data/evaluated/*/three/test/noising_orthographic_*.jsonl'
"""