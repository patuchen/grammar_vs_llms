# Example usage:

```bash
python -m main --lp cs-uk --model Unbabel/TowerInstruct-7B-v0.2 --prompt base --split micro_test --perturbation character_noise
```

Outputs translations to `../output_translations/wmt24/system-outputs/{args.lp}/{args.prompt}_{args.split}_results.jsonl` as JSONL files, ready for evaluation.

# Working models (incl. chat templates):
- Unbabel/TowerInstruct-7B-v0.2
- Unbabel/EuroLLM-1.7B-Instruct
- meta-llama/Meta-Llama-3.1-8B-Instruct
- gpt-4o-mini

# TODOs:
- add other noising functions
- add other prompts
- define system prompt for gpt
    - is "You are a helpful machine translation assistant." enough?
- Do we use our chat templates or thoses defined by openai/anthropic/vllm?
- vllm: chat vs generate
- openai/anthropic: batch mode to reduce the cost?
