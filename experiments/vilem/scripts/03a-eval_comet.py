import argparse
import csv
import glob
import numpy as np
import scipy.stats
import json

from comet import download_model, load_from_checkpoint
model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

args = argparse.ArgumentParser()
args.add_argument("--data", "-d", default="data/gpt-4o-mini_de-en_cs-uk_500/")
args = args.parse_args()

for langs in glob.glob(args.data + "/*"):
    fout = open(
        args.data.replace("data/", "computed/evals/").removesuffix("/")+f"---{langs.removeprefix(args.data)}.jsonl",
        "w"
    )
    print(langs.removeprefix(args.data)+"\n")
    scores_all = {}
    for fname in glob.glob(f"{langs}/*.csv"):
        with open(fname, "r") as f:
            data = list(csv.DictReader(f))
        fname = fname.removeprefix(f"{langs}/")

        data = [
            {
                "mt": x["translation"],
                "src": x["source"],
                "ref": x["reference"],
            }
            for x in data
        ]
        scores_all[fname] = model.predict(data, gpus=1)["scores"]

    fname_special = [x for x in scores_all.keys() if "no_errors" in x][0]
    scores_special = scores_all[fname_special]
    for fname, scores in scores_all.items():
        print(
            f"{fname:>40}",
            f"{np.average(scores):.4f}",
            f"p={scipy.stats.ttest_ind(scores, scores_special).pvalue:.5f}",
            sep=" | ",
        )
        fout.write(json.dumps({
            "fname": fname,
            "score": np.average(scores)
        })+"\n")
    fout.close()