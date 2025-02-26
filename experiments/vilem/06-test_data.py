# %%

import grammar_v_mtllm
import collections

data = grammar_v_mtllm.utils.load_data(split="micro_test", langs="en-cs")
assert len(data) == 10
data = grammar_v_mtllm.utils.load_data(split="tiny_test", langs="en-cs")
assert len(data) == 100
data = grammar_v_mtllm.utils.load_data(split="tiny_test", langs="all")
assert len(data) == 1100 # 100 per each language

# %%

data = grammar_v_mtllm.utils.load_data(langs="three")
print(len(data))

# %%

data_by_lang = collections.defaultdict(list)
data = grammar_v_mtllm.utils.load_data(split="all", langs="all")
for line in data:
    data_by_lang[line["langs"]].append(line)

print({k: len(v) for k,v in data_by_lang.items()})