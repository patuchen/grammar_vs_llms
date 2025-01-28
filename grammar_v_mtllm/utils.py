from typing import List, Union
from typing import Dict

def ensure_wmt_exists():
    import requests
    import os
    import tarfile

    if not os.path.exists("data/mt-metrics-eval-v2/"):
        print("Downloading WMT data because data/mt-metrics-eval-v2/ does not exist..")
        os.makedirs("data/", exist_ok=True)
        r = requests.get("https://storage.googleapis.com/mt-metrics-eval/mt-metrics-eval-v2.tgz")
        with open("data/mt-metrics-eval-v2.tgz", "wb") as f:
            f.write(r.content)
        with tarfile.open("data/mt-metrics-eval-v2.tgz", "r:gz") as f:
            f.extractall("data/")
        os.remove("data/mt-metrics-eval-v2.tgz")


def load_data_wmt(year="wmt23", langs="en-cs"):  # noqa: C901
    import glob
    import collections
    import numpy as np
    import os
    import pickle
    import contextlib

    # temporarily change to the root directory
    with contextlib.chdir(os.path.dirname(os.path.realpath(__file__)) + "/../"):
        ensure_wmt_exists()

        os.makedirs("data/cache/", exist_ok=True)
        cache_f = f"data/cache/{year}_{langs}.pkl"

        # load cache if exists
        if os.path.exists(cache_f):
            with open(cache_f, "rb") as f:
                return pickle.load(f)

        lines_src = open(f"data/mt-metrics-eval-v2/{year}/sources/{langs}.txt", "r").readlines()
        lines_doc = open(f"data/mt-metrics-eval-v2/{year}/documents/{langs}.docs", "r").readlines()
        lines_ref = None
        for fname in [
            f"data/mt-metrics-eval-v2/{year}/references/{langs}.refA.txt",
            f"data/mt-metrics-eval-v2/{year}/references/{langs}.refB.txt",
            f"data/mt-metrics-eval-v2/{year}/references/{langs}.refC.txt",
            f"data/mt-metrics-eval-v2/{year}/references/{langs}.refa.txt",
            f"data/mt-metrics-eval-v2/{year}/references/{langs}.refb.txt",
            f"data/mt-metrics-eval-v2/{year}/references/{langs}.refc.txt",
            f"data/mt-metrics-eval-v2/{year}/references/{langs}.ref.txt",
        ]:
            if os.path.exists(fname):
                lines_ref = open(fname, "r").readlines()
                break
        if lines_ref is None:
            return []

        # take care of canaries because scores don't have them
        if lines_src[0].lower().startswith("canary"):
            lines_src.pop(0)
        if lines_doc[0].lower().startswith("canary"):
            lines_doc.pop(0)
        if lines_ref[0].lower().startswith("canary"):
            lines_ref.pop(0)

        line_model = {}
        for f in glob.glob(f"data/mt-metrics-eval-v2/{year}/system-outputs/{langs}/*.txt"):
            model = f.split("/")[-1].removesuffix(".txt")
            if model in {"synthetic_ref", "refA", "chrf_bestmbr"}:
                continue

            line_model[model] = open(f, "r").readlines()
            if line_model[model][0].lower().startswith("canary"):
                line_model[model].pop(0)

        models = list(line_model.keys())

        lines_score = collections.defaultdict(list)
        for fname in [
            f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.esa.seg.score",
            f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.da-sqm.seg.score",
            f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.mqm.seg.score",
            f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.wmt.seg.score",
            f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.appraise.seg.score",
            f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.wmt-raw.seg.score",
            f"data/mt-metrics-eval-v2/{year}/human-scores/{langs}.wmt-appraise.seg.score",
            False
        ]:
            if fname and os.path.exists(fname):
                break

        if not fname:
            # did not find human scores
            return []

        for line_raw in open(fname, "r").readlines():
            model, score = line_raw.strip().split()
            lines_score[model].append({"human": score})

        for f in glob.glob(f"data/mt-metrics-eval-v2/{year}/metric-scores/{langs}/*-refA.seg.score"):
            metric = f.split("/")[-1].removesuffix("-refA.seg.score")
            for line_i, line_raw in enumerate(open(f, "r").readlines()):
                model, score = line_raw.strip().split("\t")
                # for refA, refB, synthetic_ref, and other "modeltems" not evaluated
                # NOTE: another option is remove the *models*
                if model not in lines_score:
                    continue
                # NOTE: there's no guarantee that this indexing is correct
                lines_score[model][line_i % len(lines_src)][metric] = float(score)

        # filter out lines that have no human score
        lines_score = {k: v for k, v in lines_score.items() if len(v) > 0}
        models = [model for model in models if model in lines_score]

        # putting it all together
        data = []
        line_id_true = 0

        # remove models that have no outputs
        models_bad = set()
        for model, scores in lines_score.items():
            if all([x["human"] in {"None", "0"} for x in scores]):
                models_bad.add(model)
        models = [model for model in models if model not in models_bad]

        for line_i, (line_src, line_ref, line_doc) in enumerate(zip(lines_src, lines_ref, lines_doc)):
            # filter None on the whole row
            # TODO: maybe still consider segments with 0?
            # NOTE: if we do that, then we won't have metrics annotations for all segments, which is bad
            if any([lines_score[model][line_i]["human"] in {"None", "0"} for model in models]):
                continue
            # metrics = set(lines_score[models[0]][line_i].keys())
            # # if we're missing some metric, skip the line
            # if any([set(lines_score[model][line_i].keys()) != metrics for model in models]):
            #     continue

            line_domain, line_doc = line_doc.strip().split("\t")

            data.append({
                "src": line_src.strip(),
                "ref": line_ref.strip(),
                "tgts": [],
                "domain": line_domain,
                "doc": line_doc,
                "src_i": line_id_true, 
                "langs": langs,
            })
            line_id_true += 1


        # save cache
        with open(cache_f, "wb") as f:
            pickle.dump(data, f)

    return data


def load_data(split, langs):
    import random

    data = {
        args: load_data_wmt(args[0], args[1])
        for args in [
            ("wmt24", "cs-uk"),
            ("wmt24", "en-cs"),
            ("wmt24", "en-de"),
            ("wmt24", "en-es"),
            ("wmt24", "en-hi"),
            ("wmt24", "en-is"),
            ("wmt24", "en-ja"),
            ("wmt24", "en-ru"),
            ("wmt24", "en-uk"),
            ("wmt24", "en-zh"),
            ("wmt24", "ja-zh"),
        ]
    }

    if langs != "all":
        if ("wmt24", langs) not in data:
            raise ValueError(f"Language pair {langs} not found in WMT24")
        data = {("wmt24", langs): data[("wmt24", langs)]}

    for k in data:
        random.Random(0).shuffle(data[k])

    if split == "micro_test":
        sample_size = 10
    elif split == "tiny_test":
        sample_size = 100
    elif split == "test":
        sample_size = 500
    
    data_new = []
    for k in data:
        data_new += data[k][:sample_size]

    return data_new