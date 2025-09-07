import re
import torch
from datasets import load_dataset, Dataset
import random
import math
import os
import sys


PATH_TO_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))





EXAMPLARS = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "cot_answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.",
        "short_answer": "6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "cot_answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
        "short_answer": "5"
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "cot_answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.",
        "short_answer": "39"
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "cot_answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.",
        "short_answer": "8"
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "cot_answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.",
        "short_answer": "9"
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "cot_answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.",
        "short_answer": "29"
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "cot_answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.",
        "short_answer": "33"
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "cot_answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.",
        "short_answer": "8"
    },
    {
        "question": "What is the largest single-digit prime number?",
        "cot_answer": "9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
        "short_answer": "7"
    }
]

# Load and prep dataset

SYSTEM_PROMPT2 = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def create_sft_data(mode):
    # Load dataset from the hub
    # dataset = load_dataset("openai/gsm8k", "main")["train"]
    if mode == "uniform":
        data = load_dataset("parquet", data_files=f"{PATH_TO_REPO}/UncertainReasoning/data/deepscaler_uniform_train.parquet")["train"]
    
    print(data[0])

def make_cot_prompt(x, num_shots=0):
    eids = list(range(len(EXAMPLARS)))
    random.shuffle(eids)
    prompts = []
    prompts.append({'role': 'system', 'content': SYSTEM_PROMPT2})
    for  i in range(num_shots):
        example = EXAMPLARS[eids[i]]
        prompts.append({'role': 'user', 'content': example["question"]})
        prompts.append({'role': 'assistant', 'content': XML_COT_FORMAT.format(
               reasoning=example["cot_answer"],
               answer=example["short_answer"]
            )})
    prompts.append({'role': 'user', 'content': x['question']})
    return prompts

def make_cot_prompt_deepscaler(x, num_shots=0):
    eids = list(range(len(EXAMPLARS)))
    random.shuffle(eids)
    prompts = []
    prompts.append({'role': 'system', 'content': SYSTEM_PROMPT})
    for i in range(num_shots):
        example = EXAMPLARS[eids[i]]
        prompts.append({'role': 'user', 'content': example["question"]})
        prompts.append({'role': 'assistant', 'content': XML_COT_FORMAT.format(
               reasoning=example["cot_answer"],
               answer=example["short_answer"]
            )})
    final_prompt = x['prompt'][0]['content'].strip() + "\n" + SYSTEM_PROMPT
    prompts.append({'role': 'user', 'content': final_prompt})
    return prompts


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_xml_reasoning(text: str) -> str:
    answer = text.split("<reasoning>")[-1]
    answer = answer.split("</reasoning>")[0]
    return answer.strip()

def extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train", num_shots=0) -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': make_cot_prompt(x, num_shots=num_shots),
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

def process_deepscaler_data(example):
    import re
    # Match and remove the instruction at the end
    pattern = r"Let's think step by step and output the final answer within \\boxed\{\}\.?\s*$"
    original_prompt = example['prompt'][0]['content']
    cleaned_prompt = re.sub(pattern, "", original_prompt).rstrip()
    example['prompt'][0]['content'] = cleaned_prompt
    return example

def get_deepscaler_questions(args, split="train", mode="uniform", num_shots=0) -> Dataset:
    data = load_dataset('parquet', data_files=f"{args.dataset_path}")[split] # type: ignore    
    # map all data['prompt][0]['content'] to remove the instruction
    data = data.map(process_deepscaler_data)
    data = data.map(lambda x: { # type: ignore
        'prompt': make_cot_prompt_deepscaler(x, num_shots=num_shots),
        'answer': x['reward_model']['ground_truth']
    }) # type: ignore
    return data # type: ignore


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # extracted_responses = [extract_boxed_answer(r) for r in responses]
    # extracted_responses = [extract_boxed_answer(r) for r in responses]
    return [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def extract_boxed_answer(text: str) -> str:
    """Extracts the full content of \boxed{...}, including nested braces."""
    start = text.find(r'\boxed{')
    if start == -1:
        return ""
    
    i = start + len(r'\boxed{')
    brace_count = 1
    content = []

    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        
        if brace_count > 0:
            content.append(text[i])
        i += 1

    return ''.join(content).strip()


def correctness_reward_bypass_template_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    rewards = []
    extracted_responses = []
    for r in responses:

        all_numbers = re.findall('\d+[.,]?\d*\s', r)
        if len(all_numbers) > 0:
            extracted_responses.append(all_numbers[-1].replace('.', '').replace(',', '').replace('\n', ''))
        else:
            extracted_responses.append(f"-1uiekc7") # no reward
            print(f"all_numbers = {all_numbers}, check this response : {r}")
         
    return [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

    
def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_boxed_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    return [
        0.5 if extract_boxed_answer(r) and r.strip().endswith(f"\\boxed{{{extract_boxed_answer(r)}}}") else 0.0
        for r in responses
    ]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if extract_boxed_answer(r) else 0.0 for r in responses]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(prompts, completions, answer, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        responses = [completion[0]['content'] for completion in completions]
        q = prompts[0][-1]['content']
        # extracted_responses = [extract_xml_answer(r) for r in responses]
        extracted_responses = [extract_boxed_answer(r) for r in responses]
        rewards = []

        for content, exanswer, sol in zip(responses, extracted_responses, answer):
            
            is_correct = False
            if exanswer==sol:
                is_correct = True
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward

# Extract final answers for Deepscaler dataset - https://github.com/limenlp/verl/blob/main/verl/utils/reward_score/math.py
def compute_score(solution_str, ground_truth) -> float:
    retval = 0.
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                retval = 1.
    except Exception as e:
        print(e)

    return retval


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


