import os
import sys
import ipdb
import pandas as pd
import diskcache as dc
from absl import app, flags
from gemba.utils import get_gemba_scores
import json
import grammar_v_mtllm
from pprint import pprint
from gemba.gpt_api import GptApi
from gemba.gemba_mqm_utils import TEMPLATE_GEMBA_MQM, apply_template, parse_mqm_answer
from gemba.gemba_esa import TEMPLATE_GEMBA_ESA_ERROR_SPANS, TEMPLATE_GEMBA_ESA_RANKING
from gemba.prompt import prompts, validate_number
from gemba.utils import CODE_MAP

flags.DEFINE_string('method', "GEMBA-MQM", 'Which prompt to use?')
flags.DEFINE_string('model', "gpt-4", 'OpenAI model')
flags.DEFINE_string('source', None, 'Filepath to the source file.')
flags.DEFINE_string('hypothesis', None, 'Filepath to the translation file.')
flags.DEFINE_string('lp', None, 'Language pair.')
# flags.DEFINE_string('perturbed_prompt', None, 'Type of prompt.')

flags.DEFINE_string('subset', "micro_test", 'Subset to use.')



def main(argv):
    FLAGS = flags.FLAGS

    data = grammar_v_mtllm.utils.load_data(split="micro_test")
    pprint(data[0])

    # answers = get_gemba_scores(source, hypothesis, FLAGS.source_lang, FLAGS.target_lang, FLAGS.method, FLAGS.model)

    cache = dc.Cache(f'cache/{FLAGS.model}_{FLAGS.method}', expire=None, size_limit=int(10e10), cull_limit=0, eviction_policy='none')
    gptapi = GptApi()

    # TODO: handle noised prompts

    # for each dict in data, and for each tgt in the dict, add a line to the df with tgt key as system, tgt value as target_seg, src as source_seg
    df = []
    for entry in data:
        for tgt, system in entry.items():
            df.append({"system": system, "source_seg": entry["src"], "target_seg": tgt, "source_lang": CODE_MAP[entry["langs"].split("-")[0]], "target_lang": CODE_MAP[entry["langs"].split("-")[1]]})

    df = pd.DataFrame(df)
    df["prompt"] = df.apply(lambda x: apply_template(prompts[FLAGS.method]['prompt'], x), axis=1)
    parse_answer = prompts[FLAGS.method]["validate_answer"]
    answers = gptapi.bulk_request(df, FLAGS.model, parse_answer, cache=cache, max_tokens=500)

    # additional step in utils
    # list(pd.DataFrame(answers)['answer'])

    for answer in answers:
        print(answer)

    # update data dict with scores
    for i, answer in enumerate(answers):
        data[i]['scores'][answer['system']][f'GEMBA_{FLAGS.model}_{FLAGS.method}'] = answer['answer']

    # save answers to file
    with open(f"scores/answers_{FLAGS.method}_{FLAGS.model}_{FLAGS.subset}.json", "w") as f:
        json.dump(answers, f)

if __name__ == "__main__":
    app.run(main)
