import argparse
import matplotlib.pyplot as plt
import numpy as np
import collections
import grammar_v_mtllm.utils_fig
import grammar_v_mtllm.utils

args = argparse.ArgumentParser()
args.add_argument("data", nargs="+")
args = args.parse_args()

data_all = grammar_v_mtllm.utils.cache_guard("beryllium", args.data)

def get_bucket_id(x):
    if not x["bucket_id"]: 
        # This is only for `llm` noise. We don't have buckets for it, so we treat 
        # every noised version of a prompt as a separate bucket.
        return x["prompt"]
    # We want to extract bucket id from something like bucket_id-1_prompt_id-mt-03-no_errors_scenario...
    return x["bucket_id"].split("-")[1].split("_")[0]

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
        data_all_joined[category][(get_bucket_id(x), x["prompt_src"])].append(x)

data_all = {
    noiser: list(noiser_v.values())
    for noiser, noiser_v in data_all_joined.items()
}

prompt_to_id = {}
def get_prompt_id(x):
    if x not in prompt_to_id:
        if x == "{source_lang}: {source_text}\\n{target_lang}:":
            prompt_to_id[x] = f"prompt minimal"
        else:
            prompt_to_id[x] = len(prompt_to_id)+1
    return prompt_to_id[x]

# each file is an individual bucket = one point
data_local = {
    noiser: [
        {
            "comet": np.average([x["eval"]["comet"] for x in data]),
            "chrf": np.average([x["eval"]["chrf"] for x in data]),
            "prompt_chrf": np.average([line["eval_prompt"]["chrf"] for line in data]),
            "prompt_ip": np.average([line["eval_prompt"]["ip"] for line in data]),
            "prompt": get_prompt_id(data[0]["prompt_src"]),
        }
        for data in noiser_v
    ]
    for noiser, noiser_v in data_all.items()
}

KEY_X = "prompt_ip"
KEY_Y = "comet"

NOISER_TO_COLOR = {
    'noising_L2': grammar_v_mtllm.utils_fig.COLORS[0],
    'noising_LazyUser': grammar_v_mtllm.utils_fig.COLORS[1],
    'noising_lexicalphrasal': grammar_v_mtllm.utils_fig.COLORS[2],
    'noising_llm': grammar_v_mtllm.utils_fig.COLORS[3],
    'noising_orthographic': grammar_v_mtllm.utils_fig.COLORS[4],
    'noising_register': grammar_v_mtllm.utils_fig.COLORS[5],
    'noising_typos_synthetic': grammar_v_mtllm.utils_fig.COLORS[6],
}
PROMPT_TO_MARKER = {
    1: ".",
    2: "x",
    3: "s",
    4: "^",
}

fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
for ax, noisers in zip(axs, [["noising_L2", "noising_LazyUser", "noising_lexicalphrasal", "noising_llm"], ["noising_orthographic", "noising_register", "noising_typos_synthetic"]]):
    for noiser in noisers:
        noiser_data = data_local[noiser]
        # disregard prompt information
        data_x = [x[KEY_X] for x in noiser_data]
        data_y = [x[KEY_Y] for x in noiser_data]
        
        for prompt in set([x["prompt"] for x in noiser_data]):
            data_x_local = [x[KEY_X] for x in noiser_data if x["prompt"] == prompt]
            data_y_local = [x[KEY_Y] for x in noiser_data if x["prompt"] == prompt]
            ax.scatter(
                data_x_local,
                data_y_local,
                marker=PROMPT_TO_MARKER[prompt],
                s=15 if prompt != 1 else 30,
                color=NOISER_TO_COLOR[noiser],
                label=grammar_v_mtllm.utils_fig.NOISER_TO_NAME[noiser] if prompt == 1 else None,
            )        
 
        ax.legend(
            ncol=1,
            handletextpad=0.2,            
        )
        ax.set_xlabel(KEY_X)
        if KEY_Y == "comet" and ax == axs[0]:
            ax.set_ylabel("Translation quality (COMET)")
        if KEY_X == "prompt_ip":
            ax.set_xlabel("Similarity to original prompt (semantic)")

    # horizontal line for minimal prompt
    ax.axhline(
        y=np.average([x[KEY_Y] for x in data_local[None]]),
        color="gray",
        linestyle="--",
    )
    ax.text(
        x=0.01,
        y=np.average([x[KEY_Y] for x in data_local[None]]),
        s="Minimal\n prompt",
        fontsize=8,
        color="#222",
        ha="left",
        va="bottom",
        # in ax coordinates
        transform=ax.transAxes,

    )
    grammar_v_mtllm.utils_fig.turn_off_spines(ax=ax)


plt.tight_layout()
plt.savefig("figures/12-beryllium.pdf")
plt.show()

        
"""
python3 experiments/src/evaluation/12-fig_beryllium.py data/evaluated/*/three/test/*.jsonl
"""