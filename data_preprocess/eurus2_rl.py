"""
Preprocess the Eurus2-RL-data dataset to parquet format

python data_preprocess/eurus2_rl.py
"""

import argparse
import datasets
import multiprocessing
import os
from jinja2 import Template
from system_prompt import GENERATOR_PROMPT_TEMPLATE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/eurus2_rl_math')
    parser.add_argument('--suffix', default='', help='Data suffix (train, test, etc.)')

    args = parser.parse_args()

    data_source = "PRIME-RL/Eurus-2-RL-Data"

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset['train']
    validation_dataset = dataset['validation']

    def make_map_fn(split):

        def process_fn(example, idx):
            question = example["prompt"][1]["content"].replace("\nPresent the answer in LaTex format: \\boxed{Your answer}", "")
            answer = example["reward_model"]["ground_truth"]
            
            prompt_template = Template(GENERATOR_PROMPT_TEMPLATE)
            prompt = prompt_template.render(prompt=question)

            if idx < 2:
                print(f"prompt: {prompt}")

            return {
                "data_source": data_source.split("/")[-1],
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'index': idx,
                    'split': split,
                    'question': question.strip(),
                }
            }
        return process_fn
    
    train_dataset = train_dataset.filter(lambda x: x["ability"] == "math")
    validation_dataset = validation_dataset.filter(lambda x: x["ability"] == "math")

    train_dataset = train_dataset.map(
        function=make_map_fn('train'), 
        with_indices=True, 
        remove_columns=train_dataset.column_names,
        num_proc=multiprocessing.cpu_count()
    )
    validation_dataset = validation_dataset.map(
        function=make_map_fn('validation'), 
        with_indices=True, 
        remove_columns=validation_dataset.column_names,
        num_proc=multiprocessing.cpu_count()
    )

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, f'train{("_" + args.suffix) if args.suffix else ""}.parquet'))
    validation_dataset.to_parquet(os.path.join(local_dir, f'validation{("_" + args.suffix) if args.suffix else ""}.parquet'))
