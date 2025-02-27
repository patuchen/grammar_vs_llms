python3 experiments/src/evaluation/03-eval_metrics.py data/translated/base_micro_test_results.jsonl

for f in data/translated/{EuroLLM,TowerInstruct}/three/test/*_scenario-orthographic_*.jsonl; do
    echo $f;
    python3 experiments/src/evaluation/03-eval_metrics.py $f --no-comet --no-ip;
done


for f in data/translated/TowerInstruct/three/*_scenario-orthographic_*.jsonl; do
    echo $f;
    python3 experiments/src/evaluation/03-eval_metrics.py $f --no-comet;
done

# sbatch_gpu_short "eval_seth-v1-eurollm" "python3 scripts/03-launch_eval_metrics.py -d data/seth-v1-eurollm/"
