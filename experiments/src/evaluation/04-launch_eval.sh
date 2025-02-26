sbatch_gpu_short "eval_seth-v1-eurollm" "python3 scripts/03-launch_eval_metrics.py -d data/seth-v1-eurollm/"
sbatch_gpu_short "eval_seth-v1-meta" "python3 scripts/03-launch_eval_metrics.py -d data/seth-v1-meta/"
sbatch_gpu_short "eval_seth-v1-tower" "python3 scripts/03-launch_eval_metrics.py -d data/seth-v1-tower/"
sbatch_gpu_short "eval_kathy-v1" "python3 scripts/03-launch_eval_metrics.py -d data/kathy-v1/"

sbatch_gpu_short "gpt-4o-mini-orig---mt-01" "python3 scripts/03-launch_eval_metrics.py -d data/gpt-4o-mini-orig/mt-01/"
sbatch_gpu_short "gpt-4o-mini-orig---mt-02" "python3 scripts/03-launch_eval_metrics.py -d data/gpt-4o-mini-orig/mt-02/"
sbatch_gpu_short "gpt-4o-mini-orig---mt-03" "python3 scripts/03-launch_eval_metrics.py -d data/gpt-4o-mini-orig/mt-03/"
sbatch_gpu_short "gpt-4o-mini-orig---mt-04" "python3 scripts/03-launch_eval_metrics.py -d data/gpt-4o-mini-orig/mt-04/"

sbatch_gpu_short "gpt-4o-mini---mt-01" "python3 scripts/03-launch_eval_metrics.py -d data/gpt-4o-mini/mt-01/"
sbatch_gpu_short "gpt-4o-mini---mt-02" "python3 scripts/03-launch_eval_metrics.py -d data/gpt-4o-mini/mt-02/"
sbatch_gpu_short "gpt-4o-mini---mt-03" "python3 scripts/03-launch_eval_metrics.py -d data/gpt-4o-mini/mt-03/"
sbatch_gpu_short "gpt-4o-mini---mt-04" "python3 scripts/03-launch_eval_metrics.py -d data/gpt-4o-mini/mt-04/"