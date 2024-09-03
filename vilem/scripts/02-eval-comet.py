import argparse
import csv
import glob

from comet import download_model, load_from_checkpoint

model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

args = argparse.ArgumentParser()
args.add_argument("--data", "-d", default="data/gpt-4o-mini_de-en_cs-uk_200/**/*.csv")
args = args.parse_args()

for fname in glob.glob(args.data):
    with open(fname, "r") as f:
        data = list(csv.DictReader(f))

    data = [
        {
            "mt": x["translation"],
            "src": x["source"],
            "ref": x["reference"],
        }
        for x in data
    ]
    score = model.predict(data, gpus=0).system_score
    print(f"{score:.4f}", fname)