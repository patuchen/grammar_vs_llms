python3 experiments/src/evaluation/03-eval_metrics.py data/translated/base_micro_test_results.jsonl

NO_COMET=1 python3 experiments/src/evaluation/03-eval_metrics.py data/translated/{EuroLLM,TowerInstruct}/three/test/*_scenario-orthographic_*.jsonl

python3 experiments/src/evaluation/03-eval_metrics.py data/translated/EuroLLM/three/test/*.jsonl
python3 experiments/src/evaluation/03-eval_metrics.py data/translated/{Llama,Qwen2.5,TowerInstruct}/three/test/*.jsonl
python3 experiments/src/evaluation/03-eval_metrics.py data/translated/GPT4o/three/test/*.jsonl
python3 experiments/src/evaluation/03-eval_metrics.py data/translated/Gemini/three/test/*.jsonl