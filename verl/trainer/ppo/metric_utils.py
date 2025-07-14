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
Metrics related to the PPO trainer.
"""

import torch
from typing import Any, Dict, List
import numpy as np
from verl import DataProto


def reduce_metrics(metrics: Dict[str, List[Any]], logging_prefix = '') -> Dict[str, Any]:
    return {f'{logging_prefix}/{key}': np.mean(val) for key, val in metrics.items()}


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True, logging_prefix: str = '') -> Dict[str, Any]:
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        f'{logging_prefix}/critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        f'{logging_prefix}/critic/score/std':
            torch.std(sequence_score).detach().item(),
        f'{logging_prefix}/critic/score/max':
            torch.max(sequence_score).detach().item(),
        f'{logging_prefix}/critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        f'{logging_prefix}/critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        f'{logging_prefix}/critic/rewards/std':
            torch.std(sequence_reward).detach().item(),
        f'{logging_prefix}/critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        f'{logging_prefix}/critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        f'{logging_prefix}/critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        f'{logging_prefix}/critic/advantages/std':
            torch.std(valid_adv).detach().item(),
        f'{logging_prefix}/critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        f'{logging_prefix}/critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        f'{logging_prefix}/critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        f'{logging_prefix}/critic/returns/std':
            torch.std(valid_returns).detach().item(),
        f'{logging_prefix}/critic/returns/max':
            torch.max(valid_returns).detach().item(),
        f'{logging_prefix}/critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            f'{logging_prefix}/critic/values/mean': torch.mean(valid_values).detach().item(),
            f'{logging_prefix}/critic/values/std': torch.std(valid_values).detach().item(),
            f'{logging_prefix}/critic/values/max': torch.max(valid_values).detach().item(),
            f'{logging_prefix}/critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            f'{logging_prefix}/critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        f'{logging_prefix}/response_length/mean':
            torch.mean(response_length).detach().item(),
        f'{logging_prefix}/response_length/std':
            torch.std(response_length).detach().item(),
        f'{logging_prefix}/response_length/max':
            torch.max(response_length).detach().item(),
        f'{logging_prefix}/response_length/min':
            torch.min(response_length).detach().item(),
        f'{logging_prefix}/response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        f'{logging_prefix}/prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        f'{logging_prefix}/prompt_length/std':
            torch.std(prompt_length).detach().item(),
        f'{logging_prefix}/prompt_length/max':
            torch.max(prompt_length).detach().item(),
        f'{logging_prefix}/prompt_length/min':
            torch.min(prompt_length).detach().item(),
        f'{logging_prefix}/prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    if 'outcome_advantages' in batch.batch and 'process_advantages' in batch.batch and 'outcome_returns' in batch.batch and 'process_returns' in batch.batch:
        outcome_advantages = batch.batch['outcome_advantages']
        process_advantages = batch.batch['process_advantages']
        outcome_returns = batch.batch['outcome_returns']
        process_returns = batch.batch['process_returns']

        outcome_advantages = torch.masked_select(outcome_advantages, response_mask)
        process_advantages = torch.masked_select(process_advantages, response_mask)
        outcome_returns = torch.masked_select(outcome_returns, response_mask)
        process_returns = torch.masked_select(process_returns, response_mask)

        metrics.update({
            f'{logging_prefix}/outcome_advantages/mean':
                torch.mean(outcome_advantages).detach().item(),
            f'{logging_prefix}/outcome_advantages/std':
                torch.std(outcome_advantages).detach().item(),
            f'{logging_prefix}/outcome_advantages/max':
                torch.max(outcome_advantages).detach().item(),
            f'{logging_prefix}/outcome_advantages/min':
                torch.min(outcome_advantages).detach().item(),
            f'{logging_prefix}/process_advantages/mean':
                torch.mean(process_advantages).detach().item(),
            f'{logging_prefix}/process_advantages/std':
                torch.std(process_advantages).detach().item(),
            f'{logging_prefix}/process_advantages/max':
                torch.max(process_advantages).detach().item(),
            f'{logging_prefix}/process_advantages/min':
                torch.min(process_advantages).detach().item(),
            f'{logging_prefix}/outcome_returns/mean':
                torch.mean(outcome_returns).detach().item(),
            f'{logging_prefix}/outcome_returns/std':
                torch.std(outcome_returns).detach().item(),
            f'{logging_prefix}/outcome_returns/max':
                torch.max(outcome_returns).detach().item(),
            f'{logging_prefix}/outcome_returns/min':
                torch.min(outcome_returns).detach().item(),
            f'{logging_prefix}/process_returns/mean':
                torch.mean(process_returns).detach().item(),
            f'{logging_prefix}/process_returns/std':
                torch.std(process_returns).detach().item(),
            f'{logging_prefix}/process_returns/max':
                torch.max(process_returns).detach().item(),
            f'{logging_prefix}/process_returns/min':
                torch.min(process_returns).detach().item(),
        })
        
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float], logging_prefix: str = '') -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'{logging_prefix}/timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'{logging_prefix}/timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int, logging_prefix: str = '') -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info['global_token_num'])
    time = timing_raw['step']
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        f'{logging_prefix}/perf/total_num_tokens': total_num_tokens,
        f'{logging_prefix}/perf/time_per_step': time,
        f'{logging_prefix}/perf/throughput': total_num_tokens / (time * n_gpus),
    }
