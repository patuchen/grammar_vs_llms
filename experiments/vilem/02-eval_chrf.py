import json
import sacrebleu
import argparse
import csv
import glob
import numpy as np
import scipy.stats
import langdetect

metric = sacrebleu.CHRF()

args = argparse.ArgumentParser()
args.add_argument("--data", "-d", default="data/gpt-4o-mini_de-en_cs-uk_500/")
args = args.parse_args()


def metric_safe(x):
    try:
        return metric.sentence_score(x["translation"], [x["reference"]]).score
    except:
        return 0


def langdetect_safe(x):
    try:
        return [
            (x.lang, x.prob)
            for x in langdetect.detect_langs(x["translation"])[:2]
        ]
    except:
        return [("unk", 1.0)]


for langs in glob.glob(args.data + "/*"):
    fout = open(
        "computed/evals_chrf/" +
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

        # for line in data:
        #     if line["translation"] == "":
        #         line["translation"] = "NULL"

        try:
            scores_out = [
                metric.sentence_score(x["translation"], [x["reference"]]).score
                for x in data
            ]
            # take top two languages
            langs_out = [
                langdetect_safe(x)
                for x in data
            ]
            data_all[fname] = (scores_out, langs_out)
        except Exception as e:
            print("ERROR in", fname, e)
            exit()

    # fname_special = [
    #     x for x in data_all.keys()
    #     if "no_errors" in x
    # ][0]
    # scores_special, _ = data_all[fname_special]
    for fname, (scores, langs) in data_all.items():
        print(
            f"{fname:>40}",
            f"{np.average(scores):.2f}",
            # f"p={scipy.stats.ttest_ind(scores, scores_special).pvalue:.5f}",
            sep=" | ",
        )
        fout.write(json.dumps({
            "fname": fname,
            "score": scores,
            "langs": langs,
        })+"\n")
    fout.close()
