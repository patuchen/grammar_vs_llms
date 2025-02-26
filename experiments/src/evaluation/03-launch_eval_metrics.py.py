import argparse
import json
import langdetect
import pathlib
from comet import download_model, load_from_checkpoint
import sacrebleu


def langdetect_safe(x):
    try:
        return [
            (x.lang, x.prob)
            for x in langdetect.detect_langs(x["translation"])[:5]
        ]
    except:
        return [("unk", 1.0)]


args = argparse.ArgumentParser()
args.add_argument("data", default="data/test.jsonl")
args.add_argument("--no-comet", action="store_true", help="Don't run COMET evaluation (useful for local no GPU environment)")
args = args.parse_args()

with open(args.data, "r") as f:
    data = [json.loads(x) for x in f]

print("Running ChrF")
metric = sacrebleu.CHRF()
scores_chrf = [
    metric.sentence_score(x["tgt"], [x["ref"]]).score
    for x in data
]

if args.no_comet:
    scores_comet = []
    print("Skipping COMET")
else:
    print("Running COMET")
    model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    scores_comet = model.predict([
        {
            "mt": x["tgt"],
            "src": x["src"],
            "ref": x["ref"],
        }
        for x in data
    ], gpus=1, batch_size=128)["scores"]

# take top 5 languages
print("Running language detection")
langs_out = [langdetect_safe(x) for x in data]

with open(f"computed/evals_comet/{pathlib.Path(args.data).stem}.jsonl", "w") as f:
    f.write(json.dumps({
        "fname": args.data,
        "score_comet": scores_comet,
        "score_chrf": scores_chrf,
        "score_langs": langs_out,
    })+"\n")