# Selective checkpoints

### Training Datasets
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
- [DeepScaleR](https://huggingface.co/datasets/lime-nlp/DeepScaleR_Difficulty)
- [GSM-Symbolic](https://github.com/apple/ml-gsm-symbolic/blob/main/generated_data/GSM_symbolic.jsonl)

### Parameters
- `mode`: Choice of algorithm. Default is `no_curr`, which means the data will be uniformly sampled. 
- `dataset_path`: Path to dataset. The dataset can be found in `data` folder.
- `L == 1200`: Required to reach the final answer.
- `gc = 8`: Use these settings to avoid out-of-memory (OOM) errors.
- `nepochs`: Number of epochs.

### Reward Formatting
The default instruction format expects the final answer to be wrapped in `\boxed{}`.  
Note:
- There are no explicit `<reasoning>` or `<answer>` tags.
- The reward function in this framework only checks for the presence of `\boxed{}` in the output.

## Setup
```bash
# Install Python
conda create -n uncertainr python=3.10
conda activate uncertainr
# Install other dependencies
pip install -r requirements.txt
```

### 1. Training
```bash
# bashfiles/run_train.sh
VLLM_USE_V1=0 python run_deepscaler.py --model_name={model_name} --dataset_path=./data/{dataset_name}_train.parquet --dataset_name={dataset_name} --mode=no_curr --uncertainty_metric=None --num_shots=0 --gc=8 --L=1200 --nepochs=1000 --T=0.0 --eta=50 --sensitivity=2.0 --beta=0.5 --d_min=0.0 --d_max=100.0 --seed=1025
```

---
### 2. Our method
#### 2.1. Training Logs

We are capturing the following information and save it every 100 steps.
- `reward`: reward per question-answer pair during training
- `extra_info`: precomputed information (difficulty, uncertainty measures)
- `average_nll`, `nll`: on-the-fly uncertainty measures, aggregated from token probabilitis during training process

```json
{
    "current_step": 1,
    "step_seen": 15,
    "logs": [
        {
            "step": ["1", "1", "1", "1", "1", "1", "1", "1"],
            "extra_info": [
                {
                    "difficulty": 2.34375, "index": 17158, "split": "train"
                },
                {
                    "difficulty": 2.34375, "index": 17158, "split": "train"
                },
                {
                    "difficulty": 2.34375, "index": 17158, "split": "train"
                },
                {
                    "difficulty": 2.34375, "index": 17158, "split": "train"
                },
                {
                    "difficulty": 2.34375, "index": 17158, "split": "train"
                },
                {
                    "difficulty": 2.34375, "index": 17158, "split": "train"
                },
                {
                    "difficulty": 2.34375, "index": 17158, "split": "train"
                },
                {
                    "difficulty": 2.34375, "index": 17158, "split": "train"
                }
            ],
            "reward": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            "average_nll": [0.1, 0.25, 0.05, 0, 0.9, 0.7, 0.5, 0.1],
            "nll": [12, 7, 20, 101, 99, 123, 444, 333],
            "batch_idx": [0, 1, 2, 3, 4, 5, 6, 7]
        }
   ]
}
```

#### 2.2. Details

For each checkpoint, we will compute the following metrics:
- **Average reward** over samples from a range of steps `[step - delta, step]` (baseline)
- **Top-p% hardest samples** (order by the uncertainty) and their average reward (our method)

After model is trained, run the following function to score all checkpoints and select the best checkpoint with highest average reward. 

```bash
# bashfile/run_scoring.sh
python run_scoring.py --base_family='Qwen' --base_model='Qwen3-0.6B' --train_data={train_data} --difficulty='average_nll' --list_eval_data math_500 minerva olympiad amc23 --save_result=True
```
This will:

- Scan all checkpoint-* folders in checkpoint_dir
- Load relevant .pt logs in training_logs/
- Save evaluation report as .json in result_dir/

--- 
### 3. Evaluation
- Another environment is needed to avoid conflict
```bash
# Install Python
conda create -n uncertainr_eval python=3.10
conda activate uncertainr_eval
# Install other dependencies
pip install -r requirements_eval.txt
```

- Using LightEval: 
```bash
# bashfiles/run_eval.sh
python run_eval.py --task=gsm8k --model_name=path/to/best/found/checkpoint/ --model_seed=model_seed --num_shots=num_shots
```

---
### 4. Generating and evaluating over validation set
```bash
# bashfiles/run_validation.sh
VLLM_USE_V1=0 python run_validation.py --model_family='Qwen' --model_name='Qwen2.5-0.5B-Instruct' --list_seed 1025 2025 3025 --n_samples={n_samples} --train_data={train_data} --metric=None
```