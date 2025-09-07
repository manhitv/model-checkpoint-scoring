module load Anaconda3
source activate
conda activate uncertainr

python run_scoring.py --base_family='Qwen' --base_model='Qwen3-0.6B' --train_data={train_data} --difficulty='average_nll' --list_eval_data math_500 minerva olympiad amc23 --save_result=True