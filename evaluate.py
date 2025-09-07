# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom evaluation tasks for LightEval."""

from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


latex_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)

expr_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(ExprExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def gsm8k_custom_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> bool:
    extracted_responses = [extract_xml_answer(r) for r in predictions][0]
    answer = formatted_doc.choices[formatted_doc.gold_index]
    answer = extract_hash_answer(answer)
    return extracted_responses == answer

def prompt_fn(line, task_name: str = None):
    """Assumes the model is either prompted to emit \\boxed{answer} or does so automatically"""
    return Doc(
        task_name=task_name,
        query=line["problem"],
        choices=[line["solution"]],
        gold_index=0,
    )

def training_prompt_fn(line, task_name: str = None):
    """Assumes the model is either prompted to emit \\boxed{answer} or does so automatically"""
    return Doc(
        task_name=task_name,
        query=line["prompt"][1]["content"],
        choices=[line["answer"]],
        gold_index=0,
    )

def easy_prompt_fn(line, task_name: str = None):
    """Assumes the model is either prompted to emit \\boxed{answer} or does so automatically"""
    return Doc(
        task_name=task_name,
        query=line["prompt"][0]["content"],
        choices=[line["reward_model"]["ground_truth"]],
        gold_index=0,
    )

def hard_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["prompt"][0]["content"],
        choices=[line["reward_model"]["ground_truth"]],
        gold_index=0,
    )


def aime_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["problem"],
        choices=[line["answer"]],
        gold_index=0,
    )

def aime25_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["problem"],
        choices=[str(line["answer"])],
        gold_index=0,
    )


def gsm8k_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[line["answer"]],
        gold_index=0,
    )

def minerva_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[line["answer"]],
        gold_index=0,
    )

def olympiad_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[line["final_answer"]],
        gold_index=0,
    )

def amc23_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["problem"],
        choices=[line["answer"]],
        gold_index=0,
    )

# Define tasks
aime24 = LightevalTaskConfig(
    name="aime24",
    suite=["custom"],
    prompt_function=aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gold_metric],
    version=1,
)
math_500 = LightevalTaskConfig(
    name="math_500",
    suite=["custom"],
    prompt_function=prompt_fn,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)
gsm8k = LightevalTaskConfig(
    name="gsm8k",
    suite=["custom"],
    prompt_function=gsm8k_prompt_fn,
    hf_repo="openai/gsm8k",
    hf_subset="main",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gold_metric],
    version=1,
)
minerva = LightevalTaskConfig(
    name="minerva",
    suite=["custom"],
    prompt_function=minerva_prompt_fn,
    hf_repo="math-ai/minervamath",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)
aime25 = LightevalTaskConfig(
    name="aime25",
    suite=["custom"],
    prompt_function=aime25_prompt_fn,
    hf_repo="MathArena/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gold_metric],
    version=1,
)
olympiad = LightevalTaskConfig(
    name="olympiad",
    suite=["custom"],
    prompt_function=olympiad_prompt_fn,
    hf_repo="Hothan/OlympiadBench",
    hf_subset="OE_MM_maths_en_COMP",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)
training = LightevalTaskConfig(
    name="training",
    suite=["custom"],
    prompt_function=training_prompt_fn,
    hf_repo="daidv1112/DIVERSE_7clusters_selected_train_samples",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)

train_0 = LightevalTaskConfig(
    name="train_0",
    suite=["custom"],
    prompt_function=training_prompt_fn,
    hf_repo="daidv1112/DIVERSE_cluster_cluster_cluster_0",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)

train_1 = LightevalTaskConfig(
    name="train_1",
    suite=["custom"],
    prompt_function=training_prompt_fn,
    hf_repo="daidv1112/DIVERSE_cluster_cluster_cluster_1",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)
train_2 = LightevalTaskConfig( 
    name="train_2",
    suite=["custom"],
    prompt_function=training_prompt_fn,
    hf_repo="daidv1112/DIVERSE_cluster_cluster_cluster_2",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)
train_3 = LightevalTaskConfig(
    name="train_3",
    suite=["custom"],
    prompt_function=training_prompt_fn,
    hf_repo="daidv1112/DIVERSE_cluster_cluster_cluster_3",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)
train_4 = LightevalTaskConfig(
    name="train_4",
    suite=["custom"],
    prompt_function=training_prompt_fn,
    hf_repo="daidv1112/DIVERSE_cluster_cluster_cluster_4",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)
train_5 = LightevalTaskConfig(
    name="train_5",
    suite=["custom"],
    prompt_function=training_prompt_fn,
    hf_repo="daidv1112/DIVERSE_cluster_cluster_cluster_5",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)
train_6 = LightevalTaskConfig(
    name="train_6",
    suite=["custom"],
    prompt_function=training_prompt_fn,
    hf_repo="daidv1112/DIVERSE_cluster_cluster_cluster_6",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)
easy_deepscaler = LightevalTaskConfig(
    name="easy_deepscaler",
    suite=["custom"],
    prompt_function=easy_prompt_fn,
    hf_repo="daidv1112/easy_deepscaler",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)
medium_deepscaler = LightevalTaskConfig(
    name="medium_deepscaler",
    suite=["custom"],
    prompt_function=easy_prompt_fn,
    hf_repo="daidv1112/medium_deepscaler",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)
hard_deepscaler = LightevalTaskConfig(
    name="hard_deepscaler",
    suite=["custom"],
    prompt_function=hard_prompt_fn,
    hf_repo="daidv1112/hard_deepscaler",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)

amc23 = LightevalTaskConfig(
    name="amc23",
    suite=["custom"],
    prompt_function=amc23_prompt_fn,
    hf_repo="knoveleng/AMC-23",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)

# Add tasks to the table
TASKS_TABLE = []
TASKS_TABLE.append(aime25)
TASKS_TABLE.append(aime24)
TASKS_TABLE.append(math_500)
TASKS_TABLE.append(gsm8k)
TASKS_TABLE.append(training)
TASKS_TABLE.append(train_0)
TASKS_TABLE.append(train_1)
TASKS_TABLE.append(train_2)
TASKS_TABLE.append(train_3)
TASKS_TABLE.append(train_4)
TASKS_TABLE.append(train_5)
TASKS_TABLE.append(train_6)
TASKS_TABLE.append(minerva)
TASKS_TABLE.append(olympiad)
TASKS_TABLE.append(easy_deepscaler)
TASKS_TABLE.append(medium_deepscaler)
TASKS_TABLE.append(hard_deepscaler)
TASKS_TABLE.append(amc23)


# MODULE LOGIC
if __name__ == "__main__":
    print([t["name"] for t in TASKS_TABLE])
    print(len(TASKS_TABLE))