. scripts/utils.sh

sbatch_gpu "eval-comet" "python3 scripts/03a-eval_comet.py"

sbatch_gpu_short "eval_seth-v1-eurollm" "python3 scripts/03a-eval_comet.py -d data/seth-v1-eurollm"
sbatch_gpu_short "eval_seth-v1-meta" "python3 scripts/03a-eval_comet.py -d data/seth-v1-meta"
sbatch_gpu_short "eval_seth-v1-tower" "python3 scripts/03a-eval_comet.py -d data/seth-v1-tower"
sbatch_gpu_short "eval_kathy-v1" "python3 scripts/03a-eval_comet.py -d data/kathy-v1"