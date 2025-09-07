import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import transformers
import tqdm
import wandb


from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig
from utils_deepscaler import *
import argparse
import random
import numpy as np
import torch.distributed as dist
from typing import Optional, Sized
from torch.utils.data import Sampler
import os
import sys
import getpass
from pathlib import Path

from reward_funcs.reward_fn import score_deepscaler

user_name = getpass.getuser()
PATH_TO_REPO = Path(f"/scratch/{user_name}/UncertainReasoning")

# Batch size for ADA-RFT (number of prompts per curriculum step)
B = 8

# ------------------------------------------------------
# 1. Callback that updates current_R_avg and current_T in memory
# ------------------------------------------------------

class CurriculumUpdateCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.trainer_ref = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.trainer_ref is None:
            return

        # Your logs contain "reward" (not "avg_reward")
        if "reward" in logs:
            trn = self.trainer_ref

            # 1) Update R_avg
            trn.current_R_avg = float(logs["reward"])

            # 2) Compute new T
            old_T = trn.current_T
            sigma = trn.sensitivity
            eta = trn.eta
            beta = trn.beta
            d_min = trn.d_min
            d_max = trn.d_max
            R_avg = trn.current_R_avg

            increment = float(eta * np.tanh(sigma * (R_avg - beta)))
            T_prime = float(np.clip(old_T + increment, d_min, d_max))
            trn.current_T = T_prime

        else:
            return


class WandbTrainingCallback(TrainerCallback):
    """
    Forwards only the standard GRPO metrics (loss, avg_reward, etc.) to WandB.
    We do NOT send current_T or current_R_avg to WandB here.
    """
    def __init__(self):
        super().__init__()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            import wandb
            wandb.log(logs)


# ------------------------------------------------------
# 2. Custom Sampler: picks top-B prompts whose difficulty is closest to T
# ------------------------------------------------------

class RepeatRandomSampler(Sampler):
    def __init__(
        self,
        data_source: Sized,
        repeat_count: int,
        trainer_ref,            # pass the entire Trainer, not just a float T
        seed: Optional[int] = None,
        batch_num: Optional[int] = B,
        is_train: Optional[bool] = None,
        mode: Optional[str] = "uniform",
    ):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.batch_num = batch_num
        self.trainer_ref = trainer_ref
        self.mode = mode
        self.seed = seed
        self.num_samples = len(data_source)
        self.is_train = is_train
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        # Each time __iter__ is called, fetch the *current* T and R_mean from the trainer:
        T = float(self.trainer_ref.current_T)
        R_mean = float(self.trainer_ref.current_R_avg)

        if self.is_train:
            if self.mode in ["adarft"]:
                difficulties = [
                    self.data_source[i]["extra_info"]["difficulty"]
                    for i in range(self.num_samples)
                ]
                sorted_idx = sorted(
                    range(self.num_samples),
                    key=lambda i: abs(difficulties[i] - T)
                )[: self.batch_num]

                indexes = [idx for idx in sorted_idx for _ in range(self.repeat_count)]
                # print("Indexes for deepscaler_skew_easy:", indexes)
                # print("Dynamic Current T:", T)
                # print("Dynamic Current R_mean:", R_mean)
                # print("=============")
                return iter(indexes)

            elif self.mode in ["no_curr"]:
                # No curriculum: just repeat all indices uniformly
                perm = torch.randperm(self.num_samples, generator=self.generator).tolist()
                indexes = [idx for idx in perm for _ in range(self.repeat_count)]
                return iter(indexes)
            

            elif self.mode == "uncertain":
                # NOTE: This is exactly like "adarft". Will fix when there is new algo.
                difficulties = [
                    self.data_source[i]["extra_info"]["difficulty"]
                    for i in range(self.num_samples)
                ]
                sorted_idx = sorted(
                    range(self.num_samples),
                    key=lambda i: abs(difficulties[i] - T)
                )[: self.batch_num]

                indexes = [idx for idx in sorted_idx for _ in range(self.repeat_count)]
                return iter(indexes)
            
            else:
                raise ValueError(f"Unknown mode: {self.mode}. Supported modes: uniform, uncertain, deepscaler_skew_easy, deepscaler_skew_difficult, deepscaler_easy_extreme, deepscaler_hard_extreme.")


        else:
            # For evaluation: purely random repeat of all indices
            perm = torch.randperm(self.num_samples, generator=self.generator).tolist()
            indexes = [idx for idx in perm for _ in range(self.repeat_count)]
            return iter(indexes)

    def __len__(self):
        return self.repeat_count * self.batch_num if self.is_train else self.num_samples * self.repeat_count 


# ------------------------------------------------------
# 3. Helper to run “generate” and return accuracy
# ------------------------------------------------------

def generate_answer(
    model,
    tokenizer,
    tokenized_samples,
    batch_size,
    max_completion_length
):
    # Exactly the same as before: run inference, return ACCURACY only.
    if dist.get_rank() == 0:
        device = model.device
        predictions = []
        generation_config = transformers.GenerationConfig(
            max_new_tokens=max_completion_length,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model.eval()
        count = len(tokenized_samples)

        status = tqdm.tqdm(tokenized_samples, desc=f"Correct: 0/{count}")
        for i in range(0, count, batch_size):
            batches = tokenized_samples[i : i + batch_size]
            with torch.inference_mode():
                longest = max(len(b[0]) for b in batches)
                padded_input_ids = torch.stack(
                    [
                        torch.tensor([tokenizer.pad_token_id] * (longest - len(ids)) + ids)
                        for ids, _ in batches
                    ]
                ).to(device)
                attn_mask = torch.stack(
                    [tokens.ne(tokenizer.pad_token_id) for tokens in padded_input_ids]
                ).to(device)

                output = model.generate(
                    input_ids=padded_input_ids,
                    attention_mask=attn_mask,
                    generation_config=generation_config,
                )

                for j, generated in enumerate(output):
                    response = tokenizer.decode(
                        generated[len(padded_input_ids[j]) :], skip_special_tokens=True
                    )
                    prediction = extract_xml_answer(response)
                    predictions.append(batches[j][1] == prediction)

                status.update(len(batches))
                status.set_description(f"Correct: {sum(predictions)}/{count}")

        return np.mean(predictions)
    return 0

def tokenize_validation(tokenizer, samples, max_prompt_length):
    tokenized_samples = []
    for sample in samples:
        prompt = sample["prompt"]
        answer = sample["answer"]
        ids = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            truncation=False,
            max_length=max_prompt_length,
        )
        tokenized_samples.append((ids, answer))
    return tokenized_samples

# ------------------------------------------------------
# 4. EvalTrainer subclass: holds current_T and current_R_avg,
#    and passes them into the sampler on every epoch.
# ------------------------------------------------------
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from typing import Any, Optional, Sized, Union
from torch import nn
from accelerate.utils import broadcast_object_list, gather, gather_object
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import pad
from collections.abc import Mapping
import pandas as pd
# _calculate_rewards is only supported in trl v0.19.0, but it might conflict with vllm & flash-attn

class EvalTrainer(GRPOTrainer):
    def __init__(self, model, processing_class, reward_funcs, training_args, train_dataset, eval_dataset, orig_args, log_dir="training_logs", log_every=100):
        # Pass only the GRPOConfig (training_args) to super()
                
        super().__init__(model=model, processing_class=processing_class, reward_funcs=reward_funcs,
                         args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
        
        self.orig_args = orig_args  # Store the original parser args for ADA-RFT
        # Copy ADA-RFT values from the original parser args (orig_args) onto the trainer itself
        self.T = float(orig_args.T)
        self.eta = float(orig_args.eta)
        self.sensitivity = float(orig_args.sensitivity)
        self.beta = float(orig_args.beta)
        self.d_min = float(orig_args.d_min)
        self.d_max = float(orig_args.d_max)

        self.current_T = self.T
        self.current_R_avg = 0.0
        
        # Selective checkpoints
        self.log_dir = Path(training_args.output_dir) / log_dir
        self._n_steps_seen = 0
        self.training_logs = []
        self.log_every_step = log_every
        os.makedirs(self.log_dir, exist_ok=True)
    
    def move_to_device(self, data): # avoid to call super()._prepare_inputs from GRPOTrainer
        if isinstance(data, Mapping):
            return type(data)({k: self.move_to_device(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self.move_to_device(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
                kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data
    
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:      
                
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        
        prompt_inputs = self.move_to_device(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # ADD ADAPTIVE LOGS - ONLY COMPLETION
        nll_per_token = -ref_per_token_logps * completion_mask  # shape: (B,)
        sum_nll = nll_per_token.sum(dim=1)
        len_completion = completion_mask.sum(dim=1)
        avg_nll = sum_nll / len_completion  # shape: (B,)
        var_logp = ref_per_token_logps.var(dim=1, unbiased=False)
        
        training_uncertainty = {
            "nll": sum_nll,
            "avg_nll": avg_nll,
            "var_logp": var_logp
        }

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        # -- SELECTIVE CHECKPOINT - Custom training logs
        training_log_per_step = {
            "step": [str(self.state.global_step)] * len(rewards),
            "extra_info": [e['extra_info'] for e in inputs],
            "reward": rewards.tolist(),
            "uncertainty": training_uncertainty,
            "batch_idx": list(range(len(rewards)))
        }
        
        self.training_logs.append(training_log_per_step)
        self._n_steps_seen += 1

        # -- Save periodically
        if self._n_steps_seen % self.log_every_step == 0 and self.training_logs:
            log_path = self.log_dir / f"log_step_{self._n_steps_seen}.pt"
            torch.save({
                "current_step": self.state.global_step,
                "step_seen": self._n_steps_seen,
                "logs": self.training_logs
            }, log_path)
            print(f"[LOG] Saved {len(self.training_logs)} samples to {log_path}")
            self.training_logs = []
        
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }  
            

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        tokenized_samples = tokenize_validation(self.processing_class, self.eval_dataset, self.args.max_prompt_length)
        eval_acc = generate_answer(
            self.model,
            self.processing_class,
            tokenized_samples,
            self.args.per_device_eval_batch_size,
            self.args.max_completion_length,
        )

        output = {
            f"{metric_key_prefix}_accuracy": eval_acc,
            "epoch": self.state.epoch,
        }

        self.log(output)
        return output

    def _get_train_sampler(self, train_dataset=None) -> Sampler:
        """
        Every time DataLoader is rebuilt (start of each epoch), pass in the latest
        current_T and current_R_avg. On epoch 1, current_R_avg == 0, so pivot = initial ada_T.
        """
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            repeat_count=self.args.num_generations,
            trainer_ref = trainer,
            seed=self.args.seed,
            batch_num=B,
            is_train=True,
            mode=self.orig_args.mode,
        )

    def _get_eval_sampler(self, eval_dataset=None) -> Sampler:
        return RepeatRandomSampler(
            data_source=self.eval_dataset,
            repeat_count=self.args.num_generations,
            trainer_ref = trainer,         
            seed=self.args.seed,
            batch_num=None,
            is_train=False,
            mode=None,  # No need for mode in eval sampler
        )

# ------------------------------------------------------
# 5. Main: parse arguments, set up trainer, and run
# ------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRPO with ADA-RFT Curriculum")
    parser.add_argument("--testing", action="store_true", help="Whether to run in testing mode", default=False)
    parser.add_argument("--model_name", type=str, required=False, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--mode", type=str, required=True, choices=["adarft", "no_curr", "uncertain"])
    parser.add_argument("--uncertainty_metric", type=str, required=True, default=None, help="The metric used to measure uncertainty")
    parser.add_argument("--dataset_path", type=str, required=True, default=None, help="Path to the dataset (if needed)")
    parser.add_argument("--num_shots", type=int, required=False, default=0)
    parser.add_argument("--nepochs", type=int, required=False, default=10) # equivalent to 10000 samples
    parser.add_argument("--seed", type=int, required=False, default=2025)
    parser.add_argument("--bs", type=int, required=False, default=1)
    parser.add_argument("--gc", type=int, required=False, default=8)
    parser.add_argument("--L", type=int, required=False, default=1200)
    parser.add_argument("--do_eval", type=int, required=False, default=0)
    parser.add_argument("--dataset_name", type=str, required=False, default="deepscaler")

    # ——— ADA-RFT hyperparameters in TEXT form ———
    parser.add_argument("--T", type=float, required=False, default=0.0, help="Initial target difficulty T ")
    parser.add_argument("--eta", type=float, required=False, default=50, help="Step size eta")
    parser.add_argument("--sensitivity", type=float, required=False, default=2.0, help="Sensitivity sigma (used inside tanh)")
    parser.add_argument("--beta", type=float, required=False, default=0.4, help="Target reward beta")
    parser.add_argument("--d_min", type=float, required=False, default=0.0, help="Lower bound on difficulty")
    parser.add_argument("--d_max", type=float, required=False, default=100.0, help="Upper bound on difficulty")

    orig_args = parser.parse_args()

    # Fix random seeds for reproducibility
    random.seed(orig_args.seed)
    torch.manual_seed(orig_args.seed)
    np.random.seed(orig_args.seed)

    # (Optional) initialize WandB for standard metrics only
    import wandb
    wandb.init(project="GRPO_training_ADARFT_text_notation", name=f"{orig_args.model_name}-shots{orig_args.num_shots}-seed{orig_args.seed}", config=vars(orig_args))

    # Only reward function is score_deepscaler
    reward_list = [score_deepscaler]

    # Load datasets (each sample must have sample["extra_info"]["difficulty"])
    train_dataset = get_deepscaler_questions(orig_args, split="train", mode=orig_args.mode)
    eval_dataset  = get_gsm8k_questions(split="test")

    # Create output directory if needed
    output_dir = f"{PATH_TO_REPO}/output/{orig_args.model_name}-{orig_args.dataset_name}-GRPO-{orig_args.num_shots}-seed{orig_args.seed}-mode{orig_args.mode}-uncertainmetric{orig_args.uncertainty_metric}-T{orig_args.T}-eta{orig_args.eta}-sensitivity{orig_args.sensitivity}-beta{orig_args.beta}-d_min{orig_args.d_min}-d_max{orig_args.d_max}"
    run_name = f"{orig_args.model_name}-GRPO-ADARFT_text_notation-shots{orig_args.num_shots}-seed{orig_args.seed}-mode{orig_args.mode}-uncertainmetric{orig_args.uncertainty_metric}-T{orig_args.T}-eta{orig_args.eta}-sensitivity{orig_args.sensitivity}-beta{orig_args.beta}-d_min{orig_args.d_min}-d_max{orig_args.d_max}"
    print("SAVING TO:", output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    do_eval = True
    eval_strategy = "steps"
    if orig_args.do_eval == 0:
        do_eval = False
        eval_strategy = "no"

    # Create GRPOConfig with only its expected fields
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        eval_strategy=eval_strategy,
        eval_steps=50,
        do_eval=do_eval,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=8,  # equals num_generations
        gradient_accumulation_steps=orig_args.gc,
        num_generations=8,
        max_prompt_length=1024,
        max_completion_length=orig_args.L,
        num_train_epochs=orig_args.nepochs,
        save_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.1,
        vllm_device="cuda:0",
        report_to="tensorboard",
        seed=orig_args.seed,
    )

    # Load the model (same as before)
    model = AutoModelForCausalLM.from_pretrained(
        orig_args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
        use_cache=False,
        offload_state_dict=False,
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(orig_args.model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    # Instantiate our EvalTrainer, passing both GRPOConfig (training_args) and parser-namespace (orig_args)
    trainer = EvalTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_list,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        orig_args=orig_args,  # give it access to T, eta, sensitivity, beta, d_min, d_max
    )
    
    # Register only the callbacks we need:
    # 1) CurriculumUpdateCallback (updates T and R_avg in memory only)
    # 2) WandbTrainingCallback   (forwards standard loss/avg_reward to WandB)
    curcallback = CurriculumUpdateCallback()
    curcallback.trainer_ref = trainer
    trainer.add_callback(curcallback)

    trainer.add_callback(WandbTrainingCallback())
    
    # Start training (curriculum updates happen purely in memory)
    trainer.train()