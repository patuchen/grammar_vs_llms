"""
Set NO_IP=1 and NO_COMET=1 to skip inner product and COMET evaluation, respectively.
"""

import argparse
import json
import os
import eval_metrics_worker
import tqdm

args = argparse.ArgumentParser()
args.add_argument("data", nargs="+")
args = args.parse_args()

for fname in tqdm.tqdm(args.data):
    with open(fname, "r") as f:
        data = [json.loads(x) for x in f]

    data = eval_metrics_worker.evaluate_data(data)

    foutname = fname.replace('/translated/', '/evaluated/')
    os.makedirs(os.path.dirname(foutname), exist_ok=True)
    with open(foutname, "w") as f:
        f.write("\n".join([json.dumps(x, ensure_ascii=False) for x in data]))