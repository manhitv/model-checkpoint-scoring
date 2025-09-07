from vllm import SamplingParams, LLM
from reward_funcs.reward_fn import score_deepscaler_validation
from transformers import AutoTokenizer
import os
import pandas as pd
import argparse

def generate_with_vllm_chat(
    prompts_text: list[list[dict]],
    model_path: str,
    sampling_params: SamplingParams,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
) -> list[str]:
    """
    Generate completions from chat-formatted prompts using vLLM.

    Args:
        prompts_chat: List of chat prompts, each a list of dicts like {"role": ..., "content": ...}
        model_path: Path or HF ID of the model
        sampling_params: vLLM SamplingParams
        tokenizer: Preloaded HF tokenizer. If None, will load from model_path.
        trust_remote_code: Whether to allow custom code for loading tokenizer/model
        dtype: "float16" or "bfloat16"
        gpu_memory_utilization: Ratio for vLLM GPU usage
        max_model_len: Max total token length (prompt + completion)

    Returns:
        completions: List of generated strings
    """
    # Initialize vLLM model
    llm = LLM(
        model=model_path,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    # Generate completions
    outputs = llm.generate(prompts_text, sampling_params)
    
    # Extract completions only (skip prompt part)
    completions = [[o.text for o in out.outputs] for out in outputs]
    return completions

def generation(model_name, input_df, seed, sampling_params, tokenizer):
    prompts_chat = input_df["prompt"].tolist()
    prompts_text = [tokenizer.apply_chat_template(p, tokenize=False) for p in prompts_chat]

    completions = generate_with_vllm_chat(
        prompts_text=prompts_text,
        model_path=model_name,
        sampling_params=sampling_params,
    )

    expected_n = sampling_params.n
    flat_completions = [c for sublist in completions for c in sublist]
    repeated_ground_truths = input_df["ground_truth"].repeat(expected_n).tolist()

    # Placeholder for actual reward function
    rewards = score_deepscaler_post_training(completions=flat_completions, answer=repeated_ground_truths)

    result_df = pd.DataFrame({
        "prompt": input_df["prompt"].repeat(expected_n).tolist(),
        "ground_truth": repeated_ground_truths,
        "completion": flat_completions,
        "reward": rewards,
        "seed": seed,
        "problem_id": input_df["problem_id"].repeat(expected_n).tolist(),
    })

    return result_df



def main_generation(list_seed, model_name, model_family, train_data, checkpoint_dir_template, df_unique, n_samples=8, generation_only=False):

    if generation_only:
        model_name = f'{model_family}/{model_name}'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        sampling_params = SamplingParams(
            max_tokens=1200, 
            stop_token_ids=[tokenizer.eos_token_id], 
            n=n_samples,
        )
        
        generated_df = generation(
            model_name=model_name, 
            input_df=df_unique, 
            seed=None, 
            sampling_params=sampling_params, 
            tokenizer=tokenizer
        )
        
        return generated_df

    list_df = []
    for seed in list_seed:
        checkpoint_dir = checkpoint_dir_template.format(
            seed=seed, 
            model_name=model_name, 
            model_family=model_family, 
            train_data=train_data
        )
        
        for checkpoint in os.listdir(checkpoint_dir):
            if 'checkpoint' in checkpoint:
                model_path = f'{checkpoint_dir}{checkpoint}'
                tokenizer = AutoTokenizer.from_pretrained(model_path)

                sampling_params = SamplingParams(
                    max_tokens=1200, 
                    stop_token_ids=[tokenizer.eos_token_id], 
                    n=n_samples,
                )
                                
                generated_df = generation(
                    model_name=model_path, 
                    input_df=df_unique, 
                    seed=seed, 
                    sampling_params=sampling_params, 
                    tokenizer=tokenizer
                )
                
                generated_df['checkpoint'] = int(checkpoint.split('-')[-1])
                generated_df['model_name'] = model_name
                generated_df['train_data'] = train_data
                print(f'[INFO] Generated model {checkpoint}')
                list_df.append(generated_df)
            
    merged_df = pd.concat(list_df)
    
    return merged_df
    
def generation_to_results(merged_df, label_df, model_name, list_seed, train_data):
    
    list_df = []
    for seed in list_seed:
        reward_df = merged_df[merged_df['seed'] == seed].groupby(['checkpoint'])['reward'].mean().reset_index()
        max_reward = reward_df['reward'].max()
        max_idx =  reward_df[reward_df['reward'] == max_reward].index[-1]
        best_checkpoint = reward_df.loc[max_idx, 'checkpoint']

        best_df = label_df[(label_df['checkpoint'] == best_checkpoint) & (label_df.model_name == model_name) 
                & (label_df.seed == seed) & (label_df.train_data == train_data)]

        list_df.append(best_df)
    
    summary = (
        pd.concat(list_df).groupby(["model_name", "task", "checkpoint"])["accuracy"]
        .agg(['mean', 'std'])
        .reset_index()
    )

    # Format as mean ± std
    summary["formatted"] = summary["mean"].round(3).astype(str) + " ± " + summary["std"].round(3).astype(str)

    # Pivot so tasks become columns
    pivot = summary.pivot(index="model_name", columns="task", values="formatted").reset_index()
    print(pivot)
    return pivot


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='LightEval')
    parser.add_argument('--model_family', type=str, required=True, default="Qwen")
    parser.add_argument('--model_name', type=str, required=True, default="Qwen2.5-0.5B-Instruct")
    parser.add_argument('--list_seed', default=[1025, 2025, 3025], nargs='+', type=int, help='List of seeds to evaluate')
    parser.add_argument('--metric', type=str, required=True, default="None")
    parser.add_argument('--n_samples', type=int, required=True, default=8) # 128 for difficulty estimation, 8 for validation set
    parser.add_argument('--train_data', type=str, required=True, default="gsm8k")
    parser.add_argument('--generation_only', type=bool, required=False, default=False)
    parser.add_argument('--data_dir', type=str, required=False, default="data/")
    parser.add_argument('--checkpoint_dir', type=str, required=False, default=None)
    parser.add_argument('--output_dir', type=str, required=False, default="selective_checkpoint/results/")
    args = parser.parse_args()

    checkpoint_dir_template = args.checkpoint_dir + '/' + '{model_family}/{model_name}-{train_data}-GRPO-0-seed{seed}-modeno_curr-uncertainmetricNone-T0.0-eta50.0-sensitivity2.0-beta0.5-d_min0.0-d_max100.0/'
    if 'train' in args.train_data: # Training mode
        df_unique = pd.read_parquet(f'{args.data_dir}/{args.train_data}_train.parquet')
    else: # validation mode
        df_unique = pd.read_parquet(f'{args.data_dir}/{args.train_data}_val.parquet')

    merged_df = main_generation(
        list_seed=args.list_seed, 
        model_name=args.model_name, 
        model_family=args.model_family, 
        train_data=args.train_data, 
        checkpoint_dir_template=checkpoint_dir_template, 
        df_unique=df_unique,
        n_samples=args.n_samples,
        generation_only=args.generation_only
    )

    merged_df.to_parquet(f'{args.output_dir}/{args.model_name}__{args.train_data}__generation.parquet')
    print(f'### Saved logging results to {args.output_dir}/')

    if not args.generation_only:
        
        from main import create_evaluation_df

        root_result_dir = f'{args.output_dir}/{args.model_family}'
        label_df = create_evaluation_df(root_result_dir=root_result_dir, base_model=args.model_name)
        
        # merge with evaluation_df
        result_df = generation_to_results(
            merged_df=merged_df,
            label_df=label_df,
            model_name=args.model_name,
            list_seed=args.list_seed,
            train_data=args.train_data
        )

        result_df.to_csv(f'{args.output_dir}/{args.model_name}__{args.train_data}__result.csv')
