import sacrebleu
import argparse
import csv
import glob
import numpy as np
import scipy.stats

metric = sacrebleu.CHRF()

args = argparse.ArgumentParser()
args.add_argument("--data", "-d", default="data/gpt-4o-mini_de-en_cs-uk_500/")
args = args.parse_args()

for langs in glob.glob(args.data + "/*"):
    print(langs.removeprefix(args.data)+"\n")
    scores_all = {}
    for fname in glob.glob(f"{langs}/*.csv"):
        with open(fname, "r") as f:
            data = list(csv.DictReader(f))
        fname = fname.removeprefix(f"{langs}/")

        scores = [
            metric.sentence_score(x["translation"], [x["reference"]]).score
            for x in data
        ]
        scores_all[fname] = scores

    fname_special = [x for x in scores_all.keys() if "polite-no_errors" in x][0]
    scores_special = scores_all[fname_special]
    for fname, scores in scores_all.items():
        print(
            f"{fname:>40}",
            f"{np.average(scores):.2f}",
            f"p={scipy.stats.ttest_ind(scores, scores_special).pvalue:.5f}",
            sep=" | ",
        )