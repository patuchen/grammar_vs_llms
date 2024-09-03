import evaluate
import argparse
import csv
import glob

metric = evaluate.load("chrf")

args = argparse.ArgumentParser()
args.add_argument("--data", "-d", default="data/gpt-4o-mini_de-en_cs-uk_200/**/*.csv")
args = args.parse_args()

for fname in glob.glob(args.data):
    with open(fname, "r") as f:
        data = list(csv.DictReader(f))

    tgt = [x["translation"] for x in data]
    ref = [x["reference"] for x in data]
    score = metric.compute(predictions=tgt, references=ref)["score"]
    print(f"{score:.2f}", fname)