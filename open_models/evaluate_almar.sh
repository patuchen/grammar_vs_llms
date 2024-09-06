#!/bin/bash
#SBATCH -J eval-mt-almar
#SBATCH -q high
#SBATCH -p gpu-troja,gpu-ms
#SBATCH -D /home/limisiewicz/my-luster/grammar_vs_llms/open_models
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=63G
#SBATCH --constraint="gpuram40G|gpuram48G"

source /home/limisiewicz/my-luster/GenderBiasGACR/.virtualenv/bin/activate


model_orig_name="haoranxu/ALMA-7B-R"
wmt_langs=("en-de")
prompt_file="../mt_base.json"


for lang_pair in "${wmt_langs[@]}"
do
  test_file="haoranxu/WMT23-Test_${lang_pair}"
  python evaluate_model.py --model_name ${model_orig_name} --test_file ${test_file} --prompt_file ${prompt_file}
done
