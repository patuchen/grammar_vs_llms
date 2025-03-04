import langdetect
import sacrebleu
import os

# load models if they're requested
if not os.environ.get("NO_IP", False):
    import sentence_transformers
    model_embd = sentence_transformers.SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
if not os.environ.get("NO_COMET", False):
    from comet import download_model, load_from_checkpoint
    model_comet = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

def langdetect_safe(x):
    try:
        return [
            (x.lang, x.prob)
            for x in langdetect.detect_langs(x["tgt"])[:5]
        ]
    except:
        return [("unk", 1.0)]

metric_chrf = sacrebleu.CHRF()

def evaluate_data(data):
    scores_chrf = [
        metric_chrf.sentence_score(x["tgt"], [x["ref"]]).score
        for x in data
    ]

    score_prompt_chrf = [
        (
            metric_chrf.sentence_score(line["prompt"], [line["prompt_src"]]).score + 
            metric_chrf.sentence_score(line["prompt_src"], [line["prompt"]]).score
        )/2
        for line in data
    ]

    if not os.environ.get("NO_IP", False):
        enc_prompt = model_embd.encode([line["prompt"] for line in data])
        enc_prompt_src = model_embd.encode([line["prompt_src"] for line in data])
        score_prompt_ip = sentence_transformers.util.cos_sim(enc_prompt, enc_prompt_src).tolist()
    else:
        score_prompt_ip = [0 for _ in data]

    if not os.environ.get("NO_COMET", False):
        scores_comet = model_comet.predict([
            {
                "mt": x["tgt"],
                "src": x["src"],
                "ref": x["ref"],
            }
            for x in data
        ], batch_size=128)["scores"]
    else:
        scores_comet = [0 for _ in data]

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
            "ip": score_prompt_ip.pop(0),
            "chrf": score_prompt_chrf.pop(0),
        }

    return data