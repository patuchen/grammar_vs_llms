---
dataset_info:
  features:
  - name: source
    dtype: string
  - name: reference
    dtype: string
  - name: model_output
    dtype: string
  - name: source_lang
    dtype: string
  - name: target_lang
    dtype: string
  - name: noised_prompt
    dtype: string
  - name: prompt
    dtype: string
  - name: noiser_type
    dtype: string
  - name: prompt_id
    dtype: string
  - name: eval_comet
    dtype: float64
  - name: eval_chrf
    dtype: float64
  - name: eval_langs
    dtype: string
  - name: eval_prompt_ip
    dtype: float64
  - name: eval_prompt_chrf
    dtype: float64
  - name: noise_level_orthographic
    dtype: float64
  - name: noise_level_lexicalphrasal
    dtype: int64
  - name: noise_level_register
    dtype: int64
  splits:
  - name: EuroLLM
    num_bytes: 402008357
    num_examples: 374724
  - name: Gemini
    num_bytes: 422951342
    num_examples: 374724
  - name: GPT4o
    num_bytes: 378853329
    num_examples: 374724
  - name: Qwen2.5
    num_bytes: 391579514
    num_examples: 374724
  - name: TowerInstruct
    num_bytes: 380345844
    num_examples: 374724
  - name: Llama
    num_bytes: 408730959
    num_examples: 374724
  download_size: 984018479
  dataset_size: 2384469345
configs:
- config_name: default
  data_files:
  - split: EuroLLM
    path: data/EuroLLM-*
  - split: Gemini
    path: data/Gemini-*
  - split: GPT4o
    path: data/GPT4o-*
  - split: Qwen2.5
    path: data/Qwen2.5-*
  - split: TowerInstruct
    path: data/TowerInstruct-*
  - split: Llama
    path: data/Llama-*
license: mit
language:
- en
- cs
- uk
- de
- zh
size_categories:
- 100K<n<1M
---

This is the dataset for our paper "How Important is 'Perfect' English for Machine Translation Prompts?"

## Dataset fields

| Key                          | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `source`                     | Source text.                                |
| `reference`                  | Reference translation in the target language.                        |
| `model_output`              | Model output|
| `source_lang`                | Language code of the source language (e.g. `cs`).                          |
| `target_lang`                | Language code of the target language (e.g. `uk`).                          |
| `prompt`                     | Clean prompt|
| `noised_prompt`              | Noised prompt |
| `noiser_type`                | Type of noise applied to the prompt (e.g. `orthographic`, `lexicalphrasal`).|
| `noise_level_orthographic`   | Level of orthographic noise, if applied (`[0,1]`).                |
| `noise_level_lexicalphrasal` | Degree of lexical or phrase-level noise, if applied (`{0,1,2}`).                      |
| `noise_level_register`       | Degree of register/style noise, if applied (`{0,1,2}`).                               |
| `prompt_id`                  | Unique noised prompt ID |
| `eval_comet`                 | COMET score for model output against reference                      |
| `eval_chrf`                  | chrF score for model output against reference.       |
| `eval_prompt_ip`             | Inner product similarity between the noised prompt and the clean prompt.       |
| `eval_prompt_chrf`           | chrF score for noised prompt against clean prompt.                                  |
| `eval_langs`                 | Language ID outputs, detected languages with confidence (e.g. `uk:0.98`).  |


## Splits

The datasets contains the following splits: `EuroLLM`, `Gemini`, `GPT4o`, `Qwen2.5`, `TowerInstruct`, and `Llama`, with outputs obtained from the corresponding model.

See the paper further details on the noise types and their implementation, model details, and language ID.

If you use this data, please cite...


