import os
import sys
import ipdb
import pandas as pd
import diskcache as dc
from absl import app, flags
from gemba.utils import get_gemba_scores
import json
import grammar_v_mtllm
from gemba.gpt_api import GptApi
from gemba.gemba_mqm_utils import TEMPLATE_GEMBA_MQM, apply_template, parse_mqm_answer
from gemba.gemba_esa import TEMPLATE_GEMBA_ESA_ERROR_SPANS, TEMPLATE_GEMBA_ESA_RANKING
from gemba.prompt import prompts, validate_number
from gemba.utils import CODE_MAP

# flags.DEFINE_string('base_prompt', "GEMBA-DA", 'prompt')
flags.DEFINE_string('model', "gpt-4", 'OpenAI model')
flags.DEFINE_string('lp', None, 'Language pair.')
flags.DEFINE_string('subset', "micro_test", 'Subset to use.')
flags.DEFINE_string('perturbation', None, 'Perturbation to use.')

def main(argv):
    FLAGS = flags.FLAGS

    data = grammar_v_mtllm.utils.load_data(split=FLAGS.subset, langs=FLAGS.lp)
    
    gptapi = GptApi()

    if FLAGS.perturbation:
        with open(f'../../noised_prompts/gemba_base_noised_{FLAGS.perturbation}.json', 'r') as file:
            prompts = json.load(file)
    else:
        with open(f'../../prompts/gemba_base.json', 'r') as file:
            prompts = json.load(file)

    # runs all prompts in file
    for prompt in prompts:
        cache = dc.Cache(f'cache/{FLAGS.model}_{FLAGS.lp}_{FLAGS.subset}_{prompt["prompt_id"]}', expire=None, size_limit=int(10e10), cull_limit=0, eviction_policy='none')

        prompt_text = prompt['prompt'] if not FLAGS.perturbation else prompt.get("noised_prompt")
        prompt_id = prompt.get("prompt_id") if not FLAGS.perturbation else prompt.get("noised_prompt_id")

        if prompt_id == "GEMBA-SQM":
            print("skipping SQM during testing")
            continue

        df = []
        for entry in data:
            for system, tgt in entry["tgt"].items():
                df.append({"system": system, "source_seg": entry["src"], "target_seg": tgt, "source_lang": CODE_MAP[entry["langs"].split("-")[0]], "target_lang": CODE_MAP[entry["langs"].split("-")[1]]})
        
        df = pd.DataFrame(df)

        df["prompt"] = df.apply(lambda x: apply_template(prompt_text, x), axis=1)
        df["prompt_id"] = prompt_id

        # parse_answer is always validate_number for our prompts
        parse_answer = validate_number

        answers = gptapi.bulk_request(df, FLAGS.model, parse_answer, cache=cache, max_tokens=100)

        # answer format:
        #     {
        #     "temperature": temperature,
        #     "answer_id": answer_id,
        #     "answer": answer,
        #     "prompt": prompt,
        #     "finish_reason": finish_reason,
        #     "model": model,
        # }

        # update data dict with scores
        for answer in answers:
            # find data entry with same source
            for j, data_entry in enumerate(data):
                if data_entry['src'] == answer['source']:
                    # data[j]['scores'][answer['system']][f'GEMBA_{FLAGS.model}_{FLAGS.base_prompt}'] = (answer['answer'], answer['answer_id'])
                    data[j]['scores'][answer['system']][f'{prompt_id}'] = (answer['answer'], answer['answer_id'])
                    # TODO: prompt information in each line - noising parameter
                    break

        for item in data:
            item["prompt_id"] = prompt_id
            item["prompt_noiser"] = prompt.get("prompt_noiser", None)
            item["prompt_p"] = prompt.get("prompt_p", None)
            item["prompt_src"] = prompt.get("prompt_src", None)
            item["noised_prompt"] = prompt.get("noised_prompt", None)

        # save answers to file, one per prompt 
        output_path = f"scores/wmt24/{FLAGS.model}/{FLAGS.lp}/{FLAGS.subset}_{prompt_id}_results.jsonl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    app.run(main)

# TODO: Correlations
# pearson segment and system
# segment - flatten all, average per segment
# system - average score per system (GEMBA) vs average human score
