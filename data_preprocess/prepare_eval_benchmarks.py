"""
Prepare evaluation benchmark datasets to parquet format

python data_preprocess/prepare_eval_benchmarks.py
"""
import argparse
import multiprocessing
from datasets import load_dataset
from jinja2 import Template
from system_prompt import GENERATOR_PROMPT_TEMPLATE

parser = argparse.ArgumentParser(description='Prepare evaluation benchmark datasets')
parser.add_argument('--datasets', nargs='+', default=['all'], 
                    choices=['all', 'MATH500', 'AIME2024', 'AIME2025', 'AMC2023', 'OlympiadBench',
                             'BGQA', 'CRUXEval', 'StrategyQA', 'TableBench'],
                    help='Datasets to process (default: all)')
parser.add_argument('--suffix', type=str, default='', help='Suffix for the dataset (default: "")')
args = parser.parse_args()

columns_to_keep = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

def generate_prompt(problem):
    prompt_template = Template(GENERATOR_PROMPT_TEMPLATE.replace("mathematical ", ""))
    prompt = prompt_template.render(prompt=problem)
    return prompt

"""
## MATH 500 ########################################################################################################################################################
"""
if 'all' in args.datasets or 'MATH500' in args.datasets:
    print("Processing MATH500 dataset...")
    MATH500_dataset = load_dataset("HuggingFaceH4/MATH-500")
    MATH500_test_dataset = MATH500_dataset["test"]
    def math_500_process(split):
        def process_fn(example, idx):
            problem = example["problem"]
            data = {
                "data_source": "MATH500",
                "prompt": [
                    {
                        "role": "user",
                        "content": generate_prompt(problem),
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example["answer"]
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': problem
                }
            }
            if idx < 2:
                print(f"Prompt {idx}: {data['prompt'][-1]['content']}\n\nAnswer: {data['reward_model']['ground_truth']}\n\n")
            return data
        
        return process_fn
    MATH500_test_dataset = MATH500_test_dataset.map(function=math_500_process('test'), with_indices=True, remove_columns=MATH500_test_dataset.column_names, num_proc=multiprocessing.cpu_count())
    MATH500_test_dataset.to_parquet(f"data/eval_benchmarks/MATH500/test{args.suffix}.parquet")
    print(f"MATH 500 test dataset size: {len(MATH500_test_dataset)}")

"""
## AIME 2024 ########################################################################################################################################################
"""
if 'all' in args.datasets or 'AIME2024' in args.datasets:
    print("Processing AIME 2024 dataset...")
    AIME2024_dataset = load_dataset("Maxwell-Jia/AIME_2024")
    AIME2024_test_dataset = AIME2024_dataset["train"]
    def AIME2024_process(split):
        def process_fn(example, idx):
            problem = example["Problem"]
            data = {
                "data_source": "AIME2024",
                "prompt": [
                    {
                        "role": "user",
                        "content": generate_prompt(problem),
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(example["Answer"])
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': problem
                }
            }
            if idx < 2:
                print(f"Prompt {idx}: {data['prompt'][-1]['content']}\n\nAnswer: {data['reward_model']['ground_truth']}\n\n")
            return data
        
        return process_fn
    AIME2024_test_dataset = AIME2024_test_dataset.map(function=AIME2024_process('test'), with_indices=True, remove_columns=AIME2024_test_dataset.column_names, num_proc=multiprocessing.cpu_count())
    AIME2024_test_dataset.to_parquet(f"data/eval_benchmarks/AIME2024/test{args.suffix}.parquet")
    print(f"AIME 2024 test dataset size: {len(AIME2024_test_dataset)}")

"""
## AIME 2025 ########################################################################################################################################################
"""
if 'all' in args.datasets or 'AIME2025' in args.datasets:
    print("Processing AIME 2025 dataset...")
    AIME2025_dataset = load_dataset("yentinglin/aime_2025")
    AIME2025_test_dataset = AIME2025_dataset["train"]
    def AIME2025_process(split):
        def process_fn(example, idx):
            problem = example["problem"]
            data = {
                "data_source": "AIME2025",
                "prompt": [
                    {
                        "role": "user",
                        "content": generate_prompt(problem),
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example["answer"]
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': problem
                }
            }
            if idx < 2:
                print(f"Prompt {idx}: {data['prompt'][-1]['content']}\n\nAnswer: {data['reward_model']['ground_truth']}\n\n")
            return data
        
        return process_fn
    AIME2025_test_dataset = AIME2025_test_dataset.map(function=AIME2025_process('test'), with_indices=True, remove_columns=AIME2025_test_dataset.column_names, num_proc=multiprocessing.cpu_count())
    AIME2025_test_dataset.to_parquet(f"data/eval_benchmarks/AIME2025/test{args.suffix}.parquet")
    print(f"AIME 2025 test dataset size: {len(AIME2025_test_dataset)}")

"""
## AMC 2023 ########################################################################################################################################################
"""
if 'all' in args.datasets or 'AMC2023' in args.datasets:
    print("Processing AMC 2023 dataset...")
    AMC2023_dataset = load_dataset("math-ai/amc23")
    AMC2023_test_dataset = AMC2023_dataset["test"]
    def AMC2023_process(split):
        def process_fn(example, idx):
            problem = example["question"]
            data = {
                "data_source": "AMC2023",
                "prompt": [
                    {
                        "role": "user",
                        "content": generate_prompt(problem),
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example["answer"]
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': problem
                }
            }
            if idx < 2:
                print(f"Prompt {idx}: {data['prompt'][-1]['content']}\n\nAnswer: {data['reward_model']['ground_truth']}\n\n")
            return data
        
        return process_fn
    AMC2023_test_dataset = AMC2023_test_dataset.map(function=AMC2023_process('test'), with_indices=True, remove_columns=AMC2023_test_dataset.column_names, num_proc=multiprocessing.cpu_count())
    AMC2023_test_dataset.to_parquet(f"data/eval_benchmarks/AMC2023/test{args.suffix}.parquet")
    print(f"AMC 2023 test dataset size: {len(AMC2023_test_dataset)}")

"""
## OlympiadBench ########################################################################################################################################################
"""
if 'all' in args.datasets or 'OlympiadBench' in args.datasets:
    print("Processing OlympiadBench dataset...")
    OlympiadBench_dataset = load_dataset("knoveleng/OlympiadBench")
    OlympiadBench_test_dataset = OlympiadBench_dataset["train"]
    def OlympiadBench_process(split):
        def process_fn(example, idx):
            problem = example["question"]
            data = {
                "data_source": "OlympiadBench",
                "prompt": [
                    {
                        "role": "user",
                        "content": generate_prompt(problem),
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example["answer"]
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': problem
                }
            }
            if idx < 2:
                print(f"Prompt {idx}: {data['prompt'][-1]['content']}\n\nAnswer: {data['reward_model']['ground_truth']}\n\n")
            return data
        
        return process_fn
    OlympiadBench_test_dataset = OlympiadBench_test_dataset.map(function=OlympiadBench_process('test'), with_indices=True, remove_columns=OlympiadBench_test_dataset.column_names, num_proc=multiprocessing.cpu_count())
    OlympiadBench_test_dataset.to_parquet(f"data/eval_benchmarks/OlympiadBench/test{args.suffix}.parquet")
    print(f"OlympiadBench test dataset size: {len(OlympiadBench_test_dataset)}")

"""
## BGQA ########################################################################################################################################################
"""
if 'all' in args.datasets or 'BGQA' in args.datasets:
    print("Processing BGQA dataset...")
    BGQA_dataset = load_dataset("tasksource/Boardgame-QA")
    BGQA_test_dataset = BGQA_dataset["test"]
    def BGQA_process(split):
        def process_fn(example, idx):
            problem = example["example"] + " Answer with True, False, or Uncertain."
            label = example["label"]
            if label == "proved":
                label = "True"
            elif label == "disproved":
                label = "False"
            else:
                label = "Uncertain"
            data = {
                "data_source": "BGQA",
                "prompt": [
                    {
                        "role": "user",
                        "content": generate_prompt(problem),
                    }
                ],
                "ability": "logic",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": label
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': problem
                }
            }
            if idx < 2:
                print(f"Prompt {idx}: {data['prompt'][-1]['content']}\n\nAnswer: {data['reward_model']['ground_truth']}\n\n")
            return data
        
        return process_fn
    BGQA_test_dataset = BGQA_test_dataset.map(function=BGQA_process('test'), with_indices=True, remove_columns=BGQA_test_dataset.column_names, num_proc=multiprocessing.cpu_count())
    BGQA_test_dataset.to_parquet(f"data/eval_benchmarks/BGQA/test{args.suffix}.parquet")
    print(f"BGQA test dataset size: {len(BGQA_test_dataset)}")

"""
## CRUXEval ########################################################################################################################################################
"""
if 'all' in args.datasets or 'CRUXEval' in args.datasets:
    print("Processing CRUXEval dataset...")
    CRUXEval_dataset = load_dataset("cruxeval-org/cruxeval", split="test")

    def CRUXEval_process(split):
        def process_fn(example, idx):
            code = example["code"]
            input = example["input"]
            output = example["output"]
            problem = f"""Given the following code:
{code}

and the following input (the order of the input is the same as the order of the input variables in the function):
{input}

Please analyze the code carefully, trace through its execution with the given input, and determine the final output."""
            

            data = {
                "data_source": "CRUXEval",
                "prompt": [
                        {
                            "role": "user",
                            "content": generate_prompt(problem),
                        }
                ],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": output
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': problem
                }
            }
            if idx < 2:
                print(f"Prompt {idx}: {data['prompt'][-1]['content']}\n\nAnswer: {data['reward_model']['ground_truth']}")
            return data
        
        return process_fn
    CRUXEval_dataset = CRUXEval_dataset.map(function=CRUXEval_process('test'), with_indices=True, remove_columns=CRUXEval_dataset.column_names, num_proc=multiprocessing.cpu_count())
    CRUXEval_dataset.to_parquet(f"data/eval_benchmarks/CRUXEval/test{args.suffix}.parquet")
    print(f"CRUXEval test dataset size: {len(CRUXEval_dataset)}")

"""
## StrategyQA ########################################################################################################################################################
"""
if 'all' in args.datasets or 'StrategyQA' in args.datasets:
    print("Processing StrategyQA dataset...")
    jsonl_path = f"./data/StrategyQA/strategyqa_train.jsonl"
    StrategyQA_dataset = load_dataset("json", data_files=jsonl_path, split="train")

    def StrategyQA_process(split):
        def process_fn(example, idx):
            problem = example["facts"] + "Based on the facts above, answer the following question. Your final answer should be either True or False.\n" + example["question"]
            gt_answer = example["answer"]
            data = {
                "data_source": "StrategyQA",
                "prompt": [
                        {
                            "role": "user",
                            "content": generate_prompt(problem),
                        }
                ],
                "ability": "commonsense",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(gt_answer)
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': problem
                }
            }
            if idx < 2:
                print(f"Prompt {idx}: {data['prompt'][-1]['content']}\n\nAnswer: {data['reward_model']['ground_truth']}\n\n")
            return data
        
        return process_fn
    StrategyQA_dataset = StrategyQA_dataset.map(function=StrategyQA_process('test'), with_indices=True, remove_columns=StrategyQA_dataset.column_names, num_proc=multiprocessing.cpu_count())
    StrategyQA_dataset.to_parquet(f"data/eval_benchmarks/StrategyQA/test{args.suffix}.parquet")
    print(f"StrategyQA test dataset size: {len(StrategyQA_dataset)}")

"""
## TableBench ########################################################################################################################################################
"""
if 'all' in args.datasets or 'TableBench' in args.datasets:
    print("Processing TableBench dataset...")
    jsonl_path = f"./data/TableBench/TableBench.jsonl"
    TableBench_dataset = load_dataset("json", data_files=jsonl_path, split="train")
    TableBench_dataset = TableBench_dataset.filter(lambda x: x["qtype"] in ["NumericalReasoning", "FactChecking"])

    def TableBench_process(split):
        def process_fn(example, idx):
            problem = f"""Read the table below in JSON format:
{example["table"]}

Based on the table, answer the following question. If your answer is extracted from the table, make sure that the answer is exactly the same as the corresponding content in the table.
{example["question"]}"""
            gt_answer = example["answer"]
            data = {
                "data_source": "TableBench",
                "prompt": [
                        {
                            "role": "user",
                            "content": generate_prompt(problem),
                        }
                ],
                "ability": "table",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": gt_answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': problem
                }
            }
            if idx < 2:
                print(f"Prompt {idx}: {data['prompt'][-1]['content']}\n\nAnswer: {data['reward_model']['ground_truth']}\n\n")
            return data

        return process_fn
    TableBench_dataset = TableBench_dataset.map(function=TableBench_process('test'), with_indices=True, remove_columns=TableBench_dataset.column_names, num_proc=multiprocessing.cpu_count())
    TableBench_dataset.to_parquet(f"data/eval_benchmarks/TableBench/test{args.suffix}.parquet")
    print(f"TableBench test dataset size: {len(TableBench_dataset)}")