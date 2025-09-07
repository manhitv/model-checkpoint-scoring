module load Anaconda3
source activate
conda activate uncertainr

VLLM_USE_V1=0 python run_deepscaler.py --model_name={model_name} --dataset_path=./data/{dataset_name}_train.parquet --dataset_name={dataset_name} --mode=no_curr --uncertainty_metric=None --num_shots=0 --gc=8 --L=1200 --nepochs=1000 --T=0.0 --eta=50 --sensitivity=2.0 --beta=0.5 --d_min=0.0 --d_max=100.0 --seed=1025