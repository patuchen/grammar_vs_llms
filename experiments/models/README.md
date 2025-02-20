# Models

Wrapper for all the models used in the experiments.

## Model interface

You can use the model loader to access the models.
`default_sampling_params()` can be used to ensure that all the models have the same parameters.


```python
model = load_model("gp4", default_sampling_params(), "You are an assistant")
```

The dictionary `MODELS` contains the names of the models and the actual models being loaded.
The models available are `gpt4`, `claude`, `eurollm` and `tower`.

The predictions can be obtained as follow:
```python
predictions = model(["<prompt_1>", "<prompt_2>", ...])
precitions
> ["<response_1>", "<response_2>", ...]
```


## API keys

We use dotenv to store the API keys so that they remain private.
The `.env` file should look like this:

```
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-other-key
```
## TODO
* Make sure that the API keys are loaded correctly
* Does it make sense to have easy-to-memorize names for the models? ("claude" instead of "claude-3-5-sonnet-20241022")
* Make sure that \_\_call\_\_ returns only the generated text
* Define the reasonable value for `default_sampling_params`
    * Update `OpenAIModel` and `AntrhopicModel` if we add new parameters
* GPT4: should we use "system" or "developer"? From the documentation, it seems the "system" has been replaced by "developer"
* See if it runs
