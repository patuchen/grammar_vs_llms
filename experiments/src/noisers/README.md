## Setup

We define 3 *classes* of noise: orthographic, lexical/phrasal, and syntactic (classes in `noising_functions.py`).
Each class may further have subtypes. For example, orthographic noise is classified into omissions, substitutions, insertions, and transpositions as observed in a spelling error corpus, as well as random typos. 
A noiser class object is instantiated by:
* `p`: The probability of noising any particular character/word/unit.
* `subtype_distribution`: A probability distribution over all these subtypes. This is user-defined, because we may be interested in different distributions depending on the situation we are trying to model.
Each noiser class has a method `noise`: str --> str.

A *scenario* defines a noise profile, which is a collection of noiser class objects. 
See examples in `utils.define_noise_profiles`. 
This allows us to model compositions of noisers.

We apply this compositions using the class `ComposeNoise` (see `noising_functions.py`). 
This is instantiated using a noise profile, and has a method `noise`: str --> str.
So we can do
```
prompt = "This is a test prompt."
profile = {
            'orthographic': {
                'p': 0.5,
                'subtype_distribution': {
                    'natural_typos': 0.03,
                    'insertion': 0.17,
                    'omission': 0.37,
                    'transposition': 0.05,
                    'substitution': 0.38
                }
            }
        }
noiser_obj = ComposeNoise(noise_profile)
noised_prompt = noiser_obj.noise(prompt)
```

## Generating noised prompts 

In `generate_and_score_noised_prompts.py`, we generate a bunch of noised prompts over a range of `p` using the noise profile `orthographic`. 
This models spelling errors+typos by L1/L2 users.

The output is saved to `../../noised_prompts/mt_base_noised_orthographic_over_p.json`.

Currently, this has 2 noised prompts per prompt and `p` (but we can generate more by setting `n_samples`).


## Next steps / TODOs

* Refine current noisers in `Orthographic`. I've written these up roughly motivated by [this paper](https://www.tandfonline.com/doi/epdf/10.1080/01434639708666335?needAccess=true), but I definitely need to go through them again. We should also eyeball the noised outputs and refine them based on that. 
* Implement noise class `Lexical/Phrasal`. This includes lexical/phrasal simplification (e.g. "provide" --> "give"), and lexical dropping ("Please translate from en to fr:" --> "translate en to fr"). The latter is motivated by search-engine like usage of LLMs by potentially lazy users (guilty). We can investigate using LLMs to help us to do these simplifications. 
* Implement noise class `WordOrder`. This reorders things in a plausible way, such as an L2 user might do. For example, "Please translate" --> "Translate please"
* Implement prompt sampling using perplexity (see below).

## Using perplexity scores
For each noised prompt, we also store its perplexity for GPT2. 
This is because we want to use perplexity as a proxy for intensity of error I. (See notes on the Overleaf.)
For niceness, we want I to be in [0,1].

For this, we need to do the following:
1) Generate a large number of prompts (high `n_samples`), covering the space of `p` when relevant. 
2) Rank them by perplexity, discard the tail using some (possibly eyeballed) threshold.
3) Sanity check: The perplexity should looks somewhat smoothly increasing with `p` i.e. it should not look like an elbow. The best case scenario is something linear.
4) Now, we treat I like a percentile. So if `I=0.5`, we return our 50th percentile prompt. 

This needs to be implemented.
