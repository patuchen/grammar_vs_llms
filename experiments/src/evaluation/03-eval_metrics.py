import argparse
import json
import langdetect
from comet import download_model, load_from_checkpoint
import sacrebleu
import sentence_transformers

def langdetect_safe(x):
    try:
        return [
            (x.lang, x.prob)
            for x in langdetect.detect_langs(x["tgt"])[:5]
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

print("Running prompt evaluation")
score_prompt_chrf = metric.sentence_score(data[0]["prompt"], [data[0]["prompt_src"]]).score
model = sentence_transformers.SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
enc_prompt = model.encode(data[0]["prompt"])
enc_prompt_src = model.encode(data[0]["prompt_src"])
score_prompt_ip = sentence_transformers.util.cos_sim(enc_prompt, enc_prompt_src).item()

if args.no_comet:
    scores_comet = [0 for _ in data]
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
    ], batch_size=128)["scores"]

print("Running language detection")
# take top 5 languages
langs_out = [langdetect_safe(x) for x in data]

# store
for line in data:
    line["eval"] = {
        "comet": scores_comet.pop(0),
        "chrf": scores_chrf.pop(0),
        "langs": langs_out.pop(0),
    }
    line["eval_prompt"] = {
        "ip": score_prompt_ip,
        "chrf": score_prompt_chrf,
        "p": data[0]["prompt_p"],
    }

with open(args.data.replace('/translated/', '/evaluated/'), "w") as f:
    f.write("\n".join([json.dumps(x) for x in data]))