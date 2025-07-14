# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Generate responses given a dataset of prompts
"""
import ray
import numpy as np
import hydra
import os
import hashlib
import json

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

import pandas as pd
from pprint import pprint
from omegaconf import OmegaConf

from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.utils import hf_tokenizer
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.model import compute_position_id_with_mask
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

# Add checkpoint path for resumable generation
def get_checkpoint_path(output_path):
    """Get the checkpoint path based on the output path."""
    output_dir = os.path.dirname(output_path)
    filename = os.path.basename(output_path)
    checkpoint_name = f".checkpoint_{filename}"
    return os.path.join(output_dir, checkpoint_name)

def get_metadata_path(output_path):
    """Get the metadata path based on the output path."""
    output_dir = os.path.dirname(output_path)
    filename = os.path.basename(output_path)
    metadata_name = f".metadata_{filename}.json"
    return os.path.join(output_dir, metadata_name)

def compute_dataset_fingerprint(dataset, prompt_key):
    """Compute a fingerprint of the dataset to ensure consistency between runs."""
    # Use the first N prompts as a dataset fingerprint
    sample_size = min(100, len(dataset))
    sample_prompts = dataset[prompt_key].iloc[:sample_size].tolist()
    
    # Convert to string and hash
    sample_str = str(sample_prompts)
    return hashlib.md5(sample_str.encode()).hexdigest()

@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get('trust_remote_code', False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()

    chat_lst = [chat.tolist() for chat in chat_lst]

    # Compute a fingerprint of the dataset
    dataset_fingerprint = compute_dataset_fingerprint(dataset, config.data.prompt_key)

    # Check for existing checkpoint
    checkpoint_path = get_checkpoint_path(config.data.output_path)
    metadata_path = get_metadata_path(config.data.output_path)
    start_batch = 0
    output_lst = [[] for _ in range(config.data.n_samples)]
    
    if os.path.exists(checkpoint_path) and os.path.exists(metadata_path):
        # Load metadata to verify dataset consistency
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        stored_fingerprint = metadata.get('dataset_fingerprint')
        
        # Verify configuration consistency
        stored_batch_size = metadata.get('batch_size')
        stored_n_samples = metadata.get('n_samples')

        config_mismatch = False
        if stored_batch_size != config.data.batch_size:
            print(f"WARNING: batch_size mismatch! Previous: {stored_batch_size}, Current: {config.data.batch_size}")
            config_mismatch = True
        
        if stored_n_samples != config.data.n_samples:
            print(f"WARNING: n_samples mismatch! Previous: {stored_n_samples}, Current: {config.data.n_samples}")
            config_mismatch = True

        if config_mismatch or stored_fingerprint != dataset_fingerprint:
            if config_mismatch:
                print(f"WARNING: Configuration mismatch! Previous: {stored_batch_size}, {stored_n_samples}, Current: {config.data.batch_size}, {config.data.n_samples}")
            else:
                print(f"WARNING: Dataset fingerprint mismatch! Previous: {stored_fingerprint}, Current: {dataset_fingerprint}")
            print("Starting from the beginning to ensure consistency.")
        else:
            print(f"Found checkpoint at {checkpoint_path}, resuming generation...")
            checkpoint_data = pd.read_parquet(checkpoint_path)
            
            # Extract the number of processed entries
            processed_count = len(checkpoint_data)
            
            # Calculate which batch to start from
            config_batch_size = config.data.batch_size
            start_batch = processed_count // config_batch_size
            
            # Load the previously generated responses
            if 'responses' in checkpoint_data.columns:
                # Convert from (n_data, n_samples) to (n_samples, n_data)
                checkpoint_responses = checkpoint_data['responses'].tolist()
                if processed_count > 0 and len(checkpoint_responses) > 0:
                    checkpoint_responses_transposed = np.array(checkpoint_responses, dtype=object).T.tolist()
                    
                    # Initialize output_lst with the existing data
                    for i, sample_responses in enumerate(checkpoint_responses_transposed):
                        if i < len(output_lst):
                            output_lst[i] = sample_responses
            
            print(f"Resuming from batch {start_batch + 1}")

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes, max_colocate_count=config.rollout.tensor_model_parallel_size)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)

    for batch_idx in range(start_batch, num_batch):
        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
        inputs = tokenizer.apply_chat_template(batch_chat_lst,
                                               add_generation_prompt=True,
                                               padding=True,
                                               truncation=True,
                                               max_length=config.rollout.prompt_length,
                                               return_tensors='pt',
                                               return_dict=True,
                                               tokenize=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

        data = DataProto.from_dict(batch_dict)
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

        # START TO GENERATE FOR n_samples TIMES
        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
        for n_sample in range(config.data.n_samples):
            output_padded = wg.generate_sequences(data_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)

            output_texts = []
            for i in range(len(output)):
                data_item = output[i]
                prompt_length = data_item.batch['prompts'].shape[-1]
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = data_item.batch['responses'][:valid_response_length]
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                output_texts.append(response_str)
                if i < 5:
                    print(response_str)

            output_lst[n_sample].extend(output_texts)
        
        # Save checkpoint after each batch
        if (batch_idx + 1) % config.get('checkpoint_frequency', 1) == 0:
            # Convert output_lst from (n_samples, n_processed_data) to (n_processed_data, n_samples)
            current_processed_count = (batch_idx + 1) * config_batch_size
            current_processed_count = min(current_processed_count, total_samples)
            
            # Transpose the output list for storage
            processed_output_lst = np.array(output_lst, dtype=object)
            processed_output_lst = np.transpose(processed_output_lst[:, :current_processed_count], axes=(1, 0)).tolist()
            
            # Create a partial dataframe and save it
            partial_dataset = dataset.iloc[:current_processed_count].copy()
            partial_dataset['responses'] = processed_output_lst
            
            # Save the checkpoint
            makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            partial_dataset.to_parquet(checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path} after batch {batch_idx + 1}")

            # Save metadata
            metadata = {
                'dataset_fingerprint': dataset_fingerprint,
                'dataset_size': len(dataset),
                'processed_count': current_processed_count,
                'timestamp': pd.Timestamp.now().isoformat(),
                'batch_size': config.data.batch_size,
                'n_samples': config.data.n_samples
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

    # add to the data frame
    dataset['responses'] = output_lst

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(config.data.output_path)
    
    # Clean up the checkpoint and metadata files when done
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Removed checkpoint file {checkpoint_path}")

    if os.path.exists(metadata_path):
        os.remove(metadata_path)
        print(f"Removed metadata file {metadata_path}")


if __name__ == '__main__':
    main()