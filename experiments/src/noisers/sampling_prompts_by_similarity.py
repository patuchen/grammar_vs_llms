'''
We get the similarities between the source prompts and noised prompts, and then sample the required number of prompts over a range of similarities.
'''
import json
import sacrebleu
from collections import defaultdict

def bucket_prompts_by_chrf(prompt_file: str, output_file: str, buckets: int):

    with open(prompt_file) as f:
        noised_prompts = json.load(f)

    # Let's get a list of base prompts
    base_prompts = list(set([prompt["prompt_id"] for prompt in noised_prompts]))
    noised_prompts_separated = {prompt_id: [] for prompt_id in base_prompts}
    for prompt in noised_prompts:
        noised_prompts_separated[prompt["prompt_id"]].append(prompt)

    # We want to bucket noised prompts per base prompt by similarity
    all_bucketed_noised_prompts = {}
    for prompt_id, noised_prompts in noised_prompts_separated.items():
        bucketed_noised_prompts = defaultdict(list)
        similarities = []
        metric = sacrebleu.CHRF()

        for prompt in noised_prompts:
            src = prompt["prompt_src"]
            tgt = prompt["noised_prompt"]
            similarity = metric.sentence_score(src, [tgt]).score
            similarities.append(similarity)
            bucketed_noised_prompts[similarity].append(prompt)
            

        # We want 10 buckets with at least 2 prompts per bucket
        buckets = 10
        bounds = (min(similarities), max(similarities))
        bucket_size = (bounds[1] - bounds[0]) / buckets
        bucketed_prompts = defaultdict(list)
        for similarity, prompts in bucketed_noised_prompts.items():
            bucket = int((similarity - bounds[0]) // bucket_size)
            bucketed_prompts[f"bucket_id_{bucket}"].extend(prompts)

        # assert len(bucketed_prompts.keys()) == buckets

        bucketed_prompts = [bucketed_prompts[f"bucket_id_{i}"] for i in range(buckets) if len(bucketed_prompts[f"bucket_id_{i}"]) > 1]

        # We'll add a bucket id to each prompt
        for idx, bucket in enumerate(bucketed_prompts):
            for prompt in bucket:
                prompt["bucket_id"] = f"bucket_id-{idx}_prompt_id-{prompt['prompt_id']}_scenario-{prompt['prompt_noiser']}" 
                # prompt["bucket_id"] = idx

        # Print number of prompts per bucket
        print(f"Prompt id: {prompt_id}")
        for idx, bucket in enumerate(bucketed_prompts):
            print(f"Bucket {idx}: {len(bucket)} prompts")

        all_bucketed_noised_prompts[prompt_id] = bucketed_prompts


    with open(output_file, "w") as f:
        json.dump(all_bucketed_noised_prompts, f, indent=4)


# # Plot correlation between bucket id and p

# import matplotlib.pyplot as plt
# import numpy as np

# X = [prompt["noise_type_parameters"]["orthographic"] for bucket in bucketed_prompts for prompt in bucket]
# Y = [prompt["bucket_id"] for bucket in bucketed_prompts for prompt in bucket]

# plt.scatter(X, Y)
# plt.xlabel("p")
# plt.ylabel("bucket_id")
# plt.savefig("bucket_id_vs_p.png")