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

flags.DEFINE_string('method', "GEMBA-MQM", 'Which prompt to use?')
flags.DEFINE_string('model', "gpt-4", 'OpenAI model')
flags.DEFINE_string('lp', None, 'Language pair.')
flags.DEFINE_string('subset', "micro_test", 'Subset to use.')


def main(argv):
    FLAGS = flags.FLAGS

    data = grammar_v_mtllm.utils.load_data(split=FLAGS.subset, langs=FLAGS.lp)

    cache = dc.Cache(f'cache/{FLAGS.model}_{FLAGS.method}_{FLAGS.lp}_{FLAGS.subset}', expire=None, size_limit=int(10e10), cull_limit=0, eviction_policy='none')
    gptapi = GptApi()

    # TODO: handle noised prompts

    df = []
    for entry in data:
        for system, tgt in entry["tgt"].items():
            df.append({"system": system, "source_seg": entry["src"], "target_seg": tgt, "source_lang": CODE_MAP[entry["langs"].split("-")[0]], "target_lang": CODE_MAP[entry["langs"].split("-")[1]]})
    print(df[-1])
    df = pd.DataFrame(df)

    # TODO: modify this to use prompt format instead of prompt file?
    df["prompt"] = df.apply(lambda x: apply_template(prompts[FLAGS.method]['prompt'], x), axis=1)

    # parse_answer is always validate_number for our prompts
    # parse_answer = prompts[FLAGS.method]["validate_answer"]
    parse_answer = validate_number

    answers = gptapi.bulk_request(df, FLAGS.model, parse_answer, cache=cache, max_tokens=500)


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
                data[j]['scores'][answer['system']][f'GEMBA_{FLAGS.model}_{FLAGS.method}'] = (answer['answer'], answer['answer_id'])
                break
        
        # data[i]['scores'][answer['system']][f'GEMBA_{FLAGS.model}_{FLAGS.method}'] = (answer['answer'], answer['temperature'])

    # save answers to file
    output_path = f"scores/{FLAGS.model}/{FLAGS.lp}/{FLAGS.method}_{FLAGS.subset}_results.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    app.run(main)
