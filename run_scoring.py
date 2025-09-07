import os
import torch
from glob import glob
import pandas as pd
import json
import re
import argparse

def get_files_in_deepest_directory(root_dir):
    max_depth = -1
    deepest_dirs = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        depth = dirpath[len(root_dir):].count(os.sep)
        if depth > max_depth:
            max_depth = depth
            deepest_dirs = [dirpath]
        elif depth == max_depth:
            deepest_dirs.append(dirpath)

    # Collect all files in the deepest directory(ies)
    deepest_files = []
    for d in deepest_dirs:
        for f in os.listdir(d):
            full_path = os.path.join(d, f)
            if os.path.isfile(full_path):
                deepest_files.append(full_path)

    return deepest_files

def get_checkpoint_step_and_root(model_path: str):
    """
    Parse model path like ".../checkpoint-400" → return (400, model_root)
    """
    base = os.path.basename(model_path)
    match = re.match(r"checkpoint-(\d+)", base)
    if not match:
        raise ValueError(f"Invalid checkpoint path: {model_path}")
    step = int(match.group(1))
    root = os.path.dirname(model_path)
    return step, root


def load_and_merge_logs_for_checkpoint(model_path: str, delta: int = 100, min_count: int = 1) -> pd.DataFrame:
    """
    Load and merge logs for a checkpoint.
    Only keep logs where step ∈ [current_step - delta, current_step]
    Only keep samples whose 'index' appears at least `min_count` times
    """
    step, model_root = get_checkpoint_step_and_root(model_path)
    log_dir = os.path.join(model_root, "training_logs")
    log_files = sorted(glob(os.path.join(log_dir, "*.pt")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_records = []
    for file in log_files:
        batch = torch.load(file, map_location=device)
        for sample in batch["logs"]:
            step_list = sample.get("step", -1)
            extra_info = sample.get('extra_info', -1)
            reward = sample.get('reward', -1)
            uncertainty = sample.get('uncertainty', {})
            if not isinstance(step_list, list):
                continue
            
            for i in range(len(step_list)): # expect batch_size
                s = int(step_list[i])
                if step - delta <= s <= step:
                    all_records.append({
                        "index": extra_info[i]["index"],
                        "difficulty": extra_info[i]["difficulty"],
                        "reward": float(reward[i]),
                        "mean_reward": sum(reward) / len(reward),
                        "step": s,
                        "nll": uncertainty['nll'][i].item(),
                        "average_nll": uncertainty['avg_nll'][i].item()
                    })

    if not all_records:
        print(f"[WARNING] No logs found in range [{step-delta}, {step}] for {model_path}")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Filter by min_count of index
    counts = df["index"].value_counts()
    keep_indices = counts[counts >= min_count].index
    df = df[df["index"].isin(keep_indices)]

    print(f"[INFO] Loaded {len(df)} logs (filtered with min_count={min_count}) for step ∈ [{step-delta}, {step}]")

    return df


def compute_metrics(df: pd.DataFrame, log_difficulty='difficulty', step=None, step_range=None, min_count=None):
    """
    Usage example:
        model_path = 'Qwen/Qwen2.5-0.5B-Instruct-GRPO-0-seed2025-modeno_curr-uncertainmetricNone-T0.0-eta50.0-sensitivity2.0-beta0.4-d_min0.0-d_max100.0'
        df = load_and_merge_logs(model_path)
        results = compute_metrics(df)
        print(results)
        log_difficulty: "difficulty", "nll", "average_nll"
    """
    results = {}
    results["step"] = step
    results["step_range"] = step_range
    results["min_count"] = min_count
    results["num_samples"] = len(df)
    reward_col = 'reward' # reward
    results["avg_reward"] = df[reward_col].mean() # baseline

    # Choose best reward by sample (index)
    df_best = df.groupby("index")[reward_col].max().reset_index() 
    df_difficulty = df.groupby("index")[log_difficulty].mean().reset_index().merge(df_best, on="index")
    reward_var = df.groupby("index")[reward_col].var()
    reward_var = reward_var[reward_var.notna()] # ignore 1 sample case
    
    # --------- Top-p percent by uncertainty ---------
    for p in list(range(1, 21)):
        top_n = max(1, int(p * len(df_difficulty) / 100))

        hardest = df_difficulty.nlargest(top_n, log_difficulty)
        results[f"top_{p}p_hard_avg"] = hardest[reward_col].mean()
        
    # ---------- Top-p percent by total reward -----
    for p in list(range(1, 21)):
        top_n = max(1, int(p * len(df_best) / 100))
        top_p_samples = df_best.nlargest(top_n, reward_col)

        # Average reward over all responses from these top-p% samples
        results[f"top_{p}p_reward_avg"] = df[df["index"].isin(top_p_samples["index"])][reward_col].mean()
        
    return results


def score_all_checkpoints(checkpoint_dir: str, uncertainty_data_path='data/deepscaler_train.parquet', result_dir: str = "selective_checkpoint/results", 
                          topk_list=[1, 10], delta=100, min_count=1, 
                          save_to_file=False, difficulty=None, log_difficulty="difficulty"
                          ):
    """
    Given a folder containing multiple checkpoint-* folders, compute metrics for each checkpoint
    and save them to a single JSON file.
    - difficulty: precomputed difficulty
    - log_difficulty: on-the-fly difficulty from logs
    """
    os.makedirs(result_dir, exist_ok=True)

    checkpoint_files = sorted(glob(os.path.join(checkpoint_dir, "checkpoint-*")))
    print(f"[INFO] Found {len(checkpoint_files)} checkpoints in {checkpoint_dir}")

    all_results = []
    all_df = []
    for checkpoint in checkpoint_files:
        try:
            df = load_and_merge_logs_for_checkpoint(checkpoint, delta=delta, min_count=min_count)
            all_df.append(df)
            
            if log_difficulty == 'difficulty':
                print('Pre uncertainty mode')
                if difficulty not in ['difficulty', 'nll', 'average_nll']:
                    raise Exception('difficulty mode is not supported')                

                uncertainty = pd.read_parquet(uncertainty_data_path)
                uncertainty['index'] = uncertainty['problem_id'].copy()

                # replace difficulty with uncertainty
                df = df.drop('difficulty', axis=1).merge(uncertainty[['index', difficulty]].rename(columns={difficulty: 'difficulty'}), on='index', how='left')
                
            if df.empty:
                print(f"[WARN] No matching logs for {checkpoint}, skipping.")
                continue
            
            step = int(df["step"].max())
            step_range_str = f"[{step - delta}, {step}]"
            result = compute_metrics(df, log_difficulty=log_difficulty, topk_list=topk_list, step_range=step_range_str, step=step, min_count=min_count)
            result["checkpoint"] = os.path.basename(checkpoint)
            result['checkpoint_dir'] = checkpoint_dir
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed to process {checkpoint}: {e}")

    # Save full results
    if save_to_file:
        timestamp = str(pd.Timestamp.now())
        file_name = f"{result_dir}/reward_summary_{timestamp}.json"
        with open(file_name, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[INFO] Saved full report to {file_name}")
    
    return all_results, all_df


def create_evaluation_df(root_result_dir, base_model):
    df = pd.DataFrame(columns=['model_name', 'checkpoint', 'task', 'seed', 'accuracy', 'start_time', 'train_data'])
    for eval_result_dir in os.listdir(root_result_dir):
        if eval_result_dir.startswith(base_model):
            seed = eval_result_dir.split('seed')[-1][:4]
            model_and_data = eval_result_dir.split('-GRPO')[0]
            if model_and_data.startswith(base_model + '-'):
                train_data = model_and_data.split(base_model + '-')[1]
            else:
                raise ValueError(f"Unexpected model_and_data format: {model_and_data}")

            eval_result_dir_full = f'{root_result_dir}/{eval_result_dir}'   
            for checkpoint_dir in os.listdir(eval_result_dir_full):
                checkpoint_dir_path = f'{eval_result_dir_full}/{checkpoint_dir}'
                files = get_files_in_deepest_directory(checkpoint_dir_path)
                for file in files:
                    if file.endswith('.json'): 
                        data = json.load(open(file, 'r'))
                        checkpoint = int(checkpoint_dir.split('-')[-1])
                        result = data['results']['all']['extractive_match']
                        start_time = data['config_general']['start_time']
                        
                        task_key = [k for k in data['config_tasks'].keys() if k.startswith('custom|')][0]
                        task_name = data['config_tasks'][task_key]['name']
                            
                        record = {
                            'model_name': base_model,
                            'checkpoint': checkpoint,
                            'train_data': train_data,
                            'task': task_name,
                            'seed': seed,
                            'accuracy': result,
                            'start_time': start_time
                        }
                        df.loc[len(df)] = record
                        
    df = df[df.model_name.str.contains(base_model)].sort_values("start_time", ascending=False)
    df_unique = df.drop_duplicates(subset=["model_name", "checkpoint", "task", "seed", 'train_data'])
    df_unique.loc[:, 'seed'] = df_unique['seed'].astype(int)
    
    return df_unique


def load_log_results(base_family, base_model, train_data, list_seed=[1025, 2025, 3025], difficulty=None, log_difficulty=None):

    output_path = '###' # modify this to your output path
    num_shot = 0
    if base_model.startswith('Llama-3.2'):
        num_shot = 1
    
    list_df = []
            
    for seed in list_seed:
        checkpoint_dir = f'{output_path}/{base_family}/{base_model}-{train_data}-GRPO-{num_shot}-seed{seed}-modeno_curr-uncertainmetricNone-T0.0-eta50.0-sensitivity2.0-beta0.5-d_min0.0-d_max100.0'
        all_results, _ = score_all_checkpoints(checkpoint_dir=checkpoint_dir, 
                                               uncertainty_data_path=f'data/{train_data}_train.parquet',
                                               result_dir='selective_checkpoint/results', topk_list=range(1, 11), delta=100, min_count=1, save_to_file=False, 
                                                    difficulty=difficulty, log_difficulty=log_difficulty)
        log_results = pd.DataFrame(all_results).sort_values(by='checkpoint', ascending=True)
        log_results['seed'] = seed
        
        list_df.append(log_results)    

    log_results = pd.concat(list_df, ignore_index=True)
    log_results['checkpoint'] = log_results['checkpoint'].apply(lambda x: int(x.split('-')[-1]) if type(x) == str else x)
    log_results = log_results[log_results.checkpoint > 0]

    return log_results

import numpy as np
def final_results(log_results, df_unique, base_model,
                  train_data,
                  log_difficulty,
                  datasets=['gsm8k', 'math_500'],
                  baseline_column='avg_reward',
                  save_result=False,
                  show_std=True):
    """
    show_std:
        True  -> show 'avg ± std'
        False -> show just avg
    Also reports number of NaN entries ignored per dataset.
    """    
    df_unique = df_unique[df_unique['train_data'] == train_data]
    good_metrics = {}
    seeds = [s for s in log_results['seed'].unique() if s != 0]

    for column in log_results.columns:
        if ('avg' in column) and (log_results[column].dtype in [np.int64, np.float64]):
            lose, tie, win = 0, 0, 0
            gap = 0
            cnt = 0

            base_per_seed = {ds: [] for ds in datasets}
            our_per_seed  = {ds: [] for ds in datasets}

            for seed in seeds:
                subset = log_results[log_results['seed'] == seed]
                ours_max = subset[column].max()
                if pd.isna(ours_max):
                    continue

                # Our best
                ours_idx = subset[subset[column] == ours_max].index[-1]
                our_ckp = subset.loc[ours_idx, 'checkpoint']

                best_checkpoint_acc = df_unique.loc[
                    (df_unique['checkpoint'] == our_ckp) &
                    (df_unique['seed'] == seed) &
                    (df_unique['task'].isin(datasets)),
                    'accuracy'
                ].sum()

                for ds in datasets:
                    acc = df_unique.loc[
                        (df_unique['checkpoint'] == our_ckp) &
                        (df_unique['seed'] == seed) &
                        (df_unique['task'] == ds),
                        'accuracy'
                    ]
                    acc_val = acc.values[0] if len(acc) > 0 else np.nan
                    our_per_seed[ds].append(acc_val)

                # Baseline best
                baseline_max = subset[baseline_column].max()
                baseline_idx = subset[subset[baseline_column] == baseline_max].index[-1]
                baseline_ckp = subset.loc[baseline_idx, 'checkpoint']

                baseline_acc_total = df_unique.loc[
                    (df_unique['checkpoint'] == baseline_ckp) &
                    (df_unique['seed'] == seed) &
                    (df_unique['task'].isin(datasets)),
                    'accuracy'
                ].sum()

                for ds in datasets:
                    acc = df_unique.loc[
                        (df_unique['checkpoint'] == baseline_ckp) &
                        (df_unique['seed'] == seed) &
                        (df_unique['task'] == ds),
                        'accuracy'
                    ]
                    acc_val = acc.values[0] if len(acc) > 0 else np.nan
                    base_per_seed[ds].append(acc_val)

                # Compare total
                if best_checkpoint_acc > baseline_acc_total:
                    win += 1
                elif best_checkpoint_acc == baseline_acc_total:
                    tie += 1
                else:
                    lose += 1

                cnt += 1
                gap += best_checkpoint_acc - baseline_acc_total

            # metrics row
            metrics_row = [lose, tie, win, round(gap * 100 / cnt, 3)]

            for ds in datasets:
                # NaN counts
                base_nan_count = np.isnan(base_per_seed[ds]).sum()
                our_nan_count  = np.isnan(our_per_seed[ds]).sum()

                # Means
                base_mean = np.nanmean(base_per_seed[ds])
                our_mean  = np.nanmean(our_per_seed[ds])

                if show_std:
                    base_std = np.nanstd(base_per_seed[ds])
                    our_std  = np.nanstd(our_per_seed[ds])
                    base_str = f"{base_mean:.3f} ± {base_std:.3f}"# (NaN:{base_nan_count})"
                    our_str  = f"{our_mean:.3f} ± {our_std:.3f}"# (NaN:{our_nan_count})"
                else:
                    base_str = f"{base_mean:.3f} (NaN:{base_nan_count})"
                    our_str  = f"{our_mean:.3f} (NaN:{our_nan_count})"

                metrics_row.extend([base_str, our_str])

            good_metrics[column] = metrics_row

    idx = ['#Lose', '#Tie', '#Win', 'Avg. Gap (%)']
    for ds in datasets:
        idx.extend([f'Base {ds.upper()}', f'Our {ds.upper()}'])

    result_df = pd.DataFrame(good_metrics, index=idx).T.sort_values(by='Avg. Gap (%)', ascending=False)

    if save_result:
        result_df.to_csv(f'selective_checkpoint/analysis/{base_model}_{train_data}__log_difficulty_{log_difficulty}__final_metrics.csv', index=True)
        print(f"Saved results to selective_checkpoint/analysis/{base_model}_{train_data}__log_difficulty_{log_difficulty}__final_metrics.csv")

    return result_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Uncertainty Scores')
    parser.add_argument('--base_family', type=str, required=True, default="Qwen")
    parser.add_argument('--base_model', type=str, required=True, default="Qwen3-0.6B")
    parser.add_argument('--train_data', type=str, required=False, default='deepscaler')
    parser.add_argument('--difficulty', type=str, required=False, default='nll', choices=['nll', 'average_nll'])
    parser.add_argument('--log_difficulty', type=str, required=False, default='difficulty', choices=['nll', 'difficulty', 'average_nll'])
    parser.add_argument('--list_eval_data', default=['gsm8k', 'math_500'], nargs='+', type=str, help='List of datasets to evaluate')
    parser.add_argument('--list_seed', default=[1025, 2025, 3025], nargs='+', type=int, help='List of seeds to evaluate')
    parser.add_argument('--save_result', default=False, type=str, help='Whether to save the result')
    parser.add_argument('--output_dir', type=str, required=False, default='###') # modify this to your output path

    args = parser.parse_args()

    root_result_dir = f'{args.output_dir}/{args.base_family}'
    label_df = create_evaluation_df(root_result_dir=root_result_dir, base_model=args.base_model)
    print('Created evaluation results for tasks: ', label_df['task'].unique(), label_df['train_data'].unique())
    
    log_results = load_log_results(base_family=args.base_family, base_model=args.base_model, train_data=args.train_data, list_seed=args.list_seed, difficulty=args.difficulty, log_difficulty=args.log_difficulty)

    result_df = final_results(log_results, label_df, train_data=args.train_data, base_model=args.base_model, datasets=args.list_eval_data, save_result=args.save_result, log_difficulty=args.log_difficulty)

    print('Calculated final metrics')
    print(result_df.head())
