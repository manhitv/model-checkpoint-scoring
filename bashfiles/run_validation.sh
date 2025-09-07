module load Anaconda3
source activate
conda activate uncertainr

VLLM_USE_V1=0 python run_validation.py --model_family='Qwen' --model_name='Qwen2.5-0.5B-Instruct' --list_seed 1025 2025 3025 --n_samples={n_samples} --train_data={train_data} --metric=None