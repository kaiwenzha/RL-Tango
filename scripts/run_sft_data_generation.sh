#!/bin/bash
set -x

data_paths=(./data/eurus2_sft_math/data.parquet)
output_paths=(./data/eurus2_sft_math/llama70b_sft_data_generation.parquet)
model_path=./base_models/Llama-3.1-70B-Instruct
prompt_length=1024
response_length=3072

for i in "${!data_paths[@]}"; do
    data_path="${data_paths[$i]}"
    output_path="${output_paths[$i]}"
    
    echo "Processing data path: $data_path"
    echo "Output path: $output_path"
    
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=4 \
        trainer.n_gpus_per_node=8 \
        data.batch_size=512 \
        data.path=$data_path \
        data.prompt_key=prompt \
        data.n_samples=1 \
        data.output_path=$output_path \
        model.path=$model_path \
        +model.trust_remote_code=True \
        rollout.temperature=0.1 \
        rollout.top_k=-1 \
        rollout.top_p=0.5 \
        rollout.prompt_length=$prompt_length \
        rollout.response_length=$response_length \
        rollout.tensor_model_parallel_size=2 \
        rollout.gpu_memory_utilization=0.4
done