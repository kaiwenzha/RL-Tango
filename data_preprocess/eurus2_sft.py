"""
Preprocess the Eurus2-SFT-Data dataset to parquet format

python data_preprocess/eurus2_sft.py
"""

import argparse
import datasets
import multiprocessing
import os
from jinja2 import Template
from system_prompt import GENERATOR_PROMPT_TEMPLATE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/eurus2_sft_math')
    parser.add_argument('--suffix', default='', help='Data suffix (train, test, etc.)')

    args = parser.parse_args()

    data_source = "PRIME-RL/Eurus-2-SFT-Data"

    dataset = datasets.load_dataset(data_source)
    
    train_dataset = dataset['train']
    train_dataset = train_dataset.filter(lambda x: x["task"] == "Math")

    def process_fn(example, idx):
        question = example["conversations"][0]["value"]
        question = question.removesuffix('\n\nPresent the answer in LaTex format: \\boxed{Your answer}')
        
        prompt_template = Template(GENERATOR_PROMPT_TEMPLATE)
        prompt = prompt_template.render(prompt=question)

        return {
            "data_source": data_source.split("/")[-1],
            "prompt": [{
                "role": "user",
                "content": prompt,
            }],
            "ability": "math",
            "extra_info": {
                'index': idx,
                'question': question.strip(),
                'dataset': example.get('dataset', '')
            }
        }
    
    processed_dataset = train_dataset.map(
        function=process_fn,
        with_indices=True, 
        remove_columns=train_dataset.column_names,
        num_proc=multiprocessing.cpu_count()
    )

    local_dir = args.local_dir
    
    os.makedirs(local_dir, exist_ok=True)

    output_file = os.path.join(local_dir, f'data{("_" + args.suffix) if args.suffix else ""}.parquet')
    processed_dataset.to_parquet(output_file)