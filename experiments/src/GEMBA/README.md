# GEMBA-MQM and GEMBA-DA

## Setup

Install required packages with python >= 3.8 

```
pip install -r requirements.txt
```

Set up secrets either for Azure API or OpenAI API: 

```
export OPENAI_AZURE_ENDPOINT=
export OPENAI_AZURE_KEY=
```

or

```
export OPENAI_API_KEY=
```

## Scoring with GEMBA

It assumes two files with the same number of lines. It prints the score for each line pair:

```
python main.py --source=source.txt --hypothesis=hypothesis.txt --source_lang=English --target_lang=Czech --method="GEMBA-MQM" --model="gpt-4"
```

The main recommended methods: `GEMBA-MQM` and `GEMBA-DA` with the model `gpt-4`.

## Collecting and evaluating experiments for GEMBA-DA

Get mt-metric-eval and download resources:

```
git clone https://github.com/google-research/mt-metrics-eval.git
cd mt-metrics-eval
pip install .
alias mtme='python3 -m mt_metrics_eval.mtme'
mtme --download
cd ..
mv ~/.mt-metrics-eval/mt-metrics-eval-v2 mt-metrics-eval-v2
```

Collect data and run the scorer

```
python gemba_da.py 

export PYTHONPATH=mt-metrics-eval:$PYTHONPATH
python evaluate.py
```

## License
GEMBA code and data are released under the [CC BY-SA 4.0 license](https://github.com/MicrosoftTranslator/GEMBA/blob/main/LICENSE.md).

## Paper
You can read more about GEMBA-DA [in our arXiv paper](https://arxiv.org/pdf/2302.14520.pdf) 
or GEMBA-MQM [in our arXiv paper](https://arxiv.org/pdf/2310.13988.pdf).

## How to Cite


### GEMBA-MQM 

    @inproceedings{kocmi-federmann-2023-gemba-mqm,
        title = {GEMBA-MQM: Detecting Translation Quality Error Spans with GPT-4},
        author = {Kocmi, Tom  and Federmann, Christian},
        booktitle = "Proceedings of the Eighth Conference on Machine Translation",
        month = dec,
        year = "2023",
        address = "Singapore",
        publisher = "Association for Computational Linguistics",
    }

### GEMBA-DA

    @inproceedings{kocmi-federmann-2023-large,
        title = "Large Language Models Are State-of-the-Art Evaluators of Translation Quality",
        author = "Kocmi, Tom and Federmann, Christian",
        booktitle = "Proceedings of the 24th Annual Conference of the European Association for Machine Translation",
        month = jun,
        year = "2023",
        address = "Tampere, Finland",
        publisher = "European Association for Machine Translation",
        url = "https://aclanthology.org/2023.eamt-1.19",
        pages = "193--203",
    }







