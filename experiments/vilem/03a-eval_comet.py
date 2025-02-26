import argparse
import csv
import glob
import numpy as np
import scipy.stats
import json
import langdetect


from comet import download_model, load_from_checkpoint
model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

args = argparse.ArgumentParser()
args.add_argument("--data", "-d", default="data/gpt-4o-mini_de-en_cs-uk_500/")
args = args.parse_args()

for langs in glob.glob(args.data + "/*"):
    fout = open(
        "computed/evals_comet/" +
        args.data.removeprefix("data/").removesuffix("/").replace("/", "---")
        + f"---{langs.removeprefix(args.data)}.jsonl",
        "w"
    )
    print(langs.removeprefix(args.data)+"\n")
    data_all = {}
    for fname in glob.glob(f"{langs}/*.csv"):
        with open(fname, "r") as f:
            data = list(csv.DictReader(f))
        fname = fname.removeprefix(f"{langs}/")

        data_comet = [
            {
                "mt": x["translation"],
                "src": x["source"],
                "ref": x["reference"],
            }
            for x in data
        ]
        # take top two languages
        langs_out = [
            [
                (x.lang, x.prob)
                for x in langdetect.detect_langs(x["translation"])[:2]
            ]
            for x in data
        ]
        scores_out = model.predict(data_comet, gpus=1)["scores"]
        data_all[fname] = (scores_out, langs_out)

    for fname, (scores, langs) in data_all.items():
        print(
            f"{fname:>40}",
            f"{np.average(scores):.4f}",
            # f"p={scipy.stats.ttest_ind(scores, scores_special).pvalue:.5f}",
            sep=" | ",
        )
        fout.write(json.dumps({
            "fname": fname,
            "score": scores,
            "langs": langs,
        })+"\n")
    fout.close()