# Grammar vs MT-LLM

Continuation of the MT Marathon 2024 project examining the effect of grammatical errors in the prompts on the quality of machine translation.
Project contributors are: Kathy Hämmerl, Patrícia Schmidtová, Vilém Zouhar, Seth Aycock, Wiktor Kamzela, Gianluca Vico, Mateusz Lango, Niyati Bafna, and Ondřej Dušek.

Not released publicly yet.

## Instructions

To access data loaders and other nice functions, you need to install the local package, which will also make sure all the dependencies are installed:
```bash
pip install -e .
```

Then, in Python, you can use the loader, which is unified across all datasets and automatically fetches data.
The first time you try to load the data, it will take long (30s-5min). Then, everything should be cached and instant:
```python
import grammar_v_mtllm
data = grammar_v_mtllm.utils.load_data(split="micro_test", langs="en-cs")
assert len(data) == 10
data = grammar_v_mtllm.utils.load_data(split="tiny_test", langs="en-cs")
assert len(data) == 100
data = grammar_v_mtllm.utils.load_data(split="tiny_test", langs="all")
assert len(data) == 1100 # 100 per each language

{x["langs"] for x in data}
> {'en-es', 'en-zh', 'en-de', 'en-hi', 'en-ru', 'en-uk', 'cs-uk', 'en-ja', 'ja-zh', 'en-cs', 'en-is'}
```

Each item in the loaded data has this structure:
```python
data[0]
> {
    "src": "Source text...",
    "ref": "Reference text...",
    # no translations available yet
    "tgts": [],
    "langs": "en-cs",
    # some additional misc, ideally don't overwrite
    "wmt": 2024,
    "src_i", 123,
    "domain": "news",
    "doc": "blesk-2024-jan04",
}
```

When you provide a translation that you wish to be evaluated, ideally add new dict to each `"tgt"` that contains all the necessary information:
```python
{
    ...
    "tgts": [
        {"tgt": "Translation by the model..", "model": "GPT4o", "prompt": "prompt v1", "perturbation": "0.1,character_noise"},
        ...
    ]
    ...
}
```


TODO: organize `experiments/`

## Contributing

- Do not commit data or executed Jupyter notebooks into this repository.
- Add your code only to `experiments/`, i.e. no new top-level directories.
- If you have any code dependencies, do not upload your `requirements.txt` but add the specific ones to `pyproject.toml`.
- Use preferably interactive Python ([works great in VSCode!](https://code.visualstudio.com/docs/python/jupyter-support-py)) over Jupyter notebooks, which are more difficult to version.