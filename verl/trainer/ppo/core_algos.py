# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict
import logging
import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_grpo_advantage(
    outcome_token_rewards: torch.Tensor,   # (B, L)  – reward on eos_mask
    eos_mask: torch.Tensor,                # (B, L)
    index: torch.Tensor,                   # (B,)    – prompt‑group id
    process_token_rewards: torch.Tensor | None = None,   # (B, n_rollouts, L)
    process_mask: torch.Tensor | None = None,            # (B, n_rollouts, L)
    outcome_weight: float = 1.0,
    process_weight: float = 1.0,
    epsilon: float = 1e-6,
    alpha: float | None = None,
):

    bsz = outcome_token_rewards.shape[0]
    device = outcome_token_rewards.device
    
    ### 1) ---------- outcome part  --------------- ###
    outcome_score = outcome_token_rewards.sum(-1)          # (B,)

    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    for i in range(bsz):
        id2score[index[i]].append(outcome_score[i])
    

    for i, (gid, vals) in enumerate(id2score.items()):
        vals = torch.stack(vals)
        id2mean[gid] = vals.mean() if vals.numel() > 1 else torch.tensor(0.0, device=device)
        id2std [gid] = vals.std()  if vals.numel() > 1 else torch.tensor(1.0, device=device)

    outcome_z = torch.empty_like(outcome_score)
    for i in range(bsz):
        gid = index[i]
        outcome_z[i] = (outcome_score[i] - id2mean[gid]) / (id2std[gid] + epsilon)

    outcome_tensor = torch.zeros_like(outcome_token_rewards)
    last_token_idx = eos_mask.sum(dim=-1) - 1
    outcome_tensor[torch.arange(bsz, device=device), last_token_idx] = outcome_z

    outcome_mask = torch.zeros_like(eos_mask, dtype=torch.bool)
    outcome_mask[torch.arange(bsz, device=device), last_token_idx] = True
    outcome_returns = outcome_tensor.flip(-1).cumsum(-1).flip(-1)
    outcome_advantages = outcome_returns.clone()

    if (
        process_token_rewards is None
        or process_mask is None
        or not process_mask.any()
        or process_weight == 0.0
    ):
        return outcome_advantages, outcome_returns

    id2mean_p, id2std_p = {}, {}
    for i, gid in enumerate(id2score.keys()):
        seq_mask = torch.tensor(index == gid, device=device)                       
        flat_vals  = process_token_rewards[seq_mask][process_mask[seq_mask]]
        if flat_vals.numel() <= 1:
            id2mean_p[gid] = torch.tensor(0.0, device=device)
            id2std_p [gid] = torch.tensor(1.0, device=device)
        else:
            id2mean_p[gid] = flat_vals.mean()
            id2std_p [gid] = flat_vals.std()

    process_tensor = torch.zeros_like(process_token_rewards)    # (B, n_rollouts, L)
    for i in range(bsz):
        gid = index[i]
        m, s = id2mean_p[gid], id2std_p[gid] + epsilon
        row_mask = process_mask[i]   # (n_rollouts, L)
        process_tensor[i, row_mask] = (process_token_rewards[i, row_mask] - m) / s
        
    ### 3) ---------- combine ------------------------ ###
    process_returns = process_tensor.flip(-1).cumsum(-1).flip(-1)    # (B, n_rollouts, L)
    process_returns = process_returns.mean(dim=1)    # (B, L)
    process_advantages = process_returns.clone() # (B, n_rollouts, L)

    if alpha is not None:
        advantages = (1 - alpha) * outcome_advantages + alpha * process_advantages
        returns = (1 - alpha) * outcome_returns + alpha * process_returns
        return advantages, returns, (1 - alpha) * outcome_advantages, (1 - alpha) * outcome_returns, alpha * process_advantages, alpha * process_returns
    else:
        advantages = outcome_weight * outcome_advantages + process_weight * process_advantages
        returns = outcome_weight * outcome_returns + process_weight * process_returns
        return advantages, returns, outcome_weight * outcome_advantages, outcome_weight * outcome_returns, process_weight * process_advantages, process_weight * process_returns
    

def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_rloo_advantage(
    outcome_token_rewards: torch.Tensor,   # (B, L)  – reward on eos_mask
    eos_mask: torch.Tensor,                # (B, L)
    index: torch.Tensor,                   # (B,)    – prompt‑group id
    process_token_rewards: torch.Tensor | None = None,   # (B, n_rollouts, L)
    process_mask: torch.Tensor | None = None,            # (B, n_rollouts, L)
    outcome_weight: float = 1.0,
    process_weight: float = 1.0,
    epsilon: float = 1e-6,
    alpha: float | None = None,
):
    bsz = outcome_token_rewards.shape[0]
    device = outcome_token_rewards.device
    
    ### 1) ---------- outcome part --------------- ###
    outcome_score = outcome_token_rewards.sum(-1)          # (B,)

    id2score = defaultdict(list)
    id2mean = {}

    for i in range(bsz):
        id2score[index[i]].append(outcome_score[i])
    

    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0, device=device)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor(id2score[idx], device=device))
        else:
            raise ValueError(f"no score in prompt index: {idx}")
    
    outcome_z = torch.empty_like(outcome_score)
    for i in range(bsz):
        gid = index[i]
        response_num = len(id2score[gid])
        if response_num > 1:
            outcome_z[i] = outcome_score[i] * response_num / (response_num - 1) - id2mean[gid] * response_num / (response_num - 1)
        else:
            outcome_z[i] = torch.tensor(0.0, device=device)
            
    outcome_tensor = torch.zeros_like(outcome_token_rewards)
    last_token_idx = eos_mask.sum(dim=-1) - 1
    outcome_tensor[torch.arange(bsz, device=device), last_token_idx] = outcome_z

    outcome_mask = torch.zeros_like(eos_mask, dtype=torch.bool)
    outcome_mask[torch.arange(bsz, device=device), last_token_idx] = True
    outcome_returns = outcome_tensor.flip(-1).cumsum(-1).flip(-1)
    outcome_advantages = outcome_returns.clone()
    

    if (
        process_token_rewards is None
        or process_mask is None
        or not process_mask.any()
        or process_weight == 0.0
    ):
        return outcome_advantages, outcome_returns

    ### 2) ---------- process part  --------------- ###
    id2process_score = defaultdict(list)
    id2process_mean = {}
    
    for i in range(bsz):
        gid = index[i]
        row_mask = process_mask[i]  # (n_rollouts, L)
        if row_mask.any():
            rewards = process_token_rewards[i, row_mask]
            id2process_score[gid].extend(rewards)

    
    for i, gid in enumerate(id2process_score.keys()):
        if len(id2process_score[gid]) == 1:
            id2process_mean[gid] = torch.tensor(0.0, device=device)
        elif len(id2process_score[gid]) > 1:
            all_rewards = torch.tensor(id2process_score[gid], device=device)
            id2process_mean[gid] = all_rewards.mean()

        else:
            raise ValueError(f"No process rewards for group {gid}")
    
    process_tensor = torch.zeros_like(process_token_rewards)    # (B, n_rollouts, L)
    for i in range(bsz):
        gid = index[i]
        row_mask = process_mask[i] # (n_rollouts, L)
        
        if not row_mask.any():
            continue
            
        response_num = len(id2process_score[gid])
        if response_num > 1:
            scale_factor = response_num / (response_num - 1)
            process_tensor[i, row_mask] = process_token_rewards[i, row_mask] * scale_factor - id2process_mean[gid] * scale_factor
        

    ### 3) ---------- combine ------------------------ ###
    process_returns = process_tensor.flip(-1).cumsum(-1).flip(-1)    # (B, n_rollouts, L)
    process_returns = process_returns.mean(dim=1)    # (B, L)
    process_advantages = 10 * process_returns.clone() # hardcoded according to observed scale

    if alpha is not None:
        advantages = (1 - alpha) * outcome_advantages + alpha * process_advantages
        returns = (1 - alpha) * outcome_returns + alpha * process_returns
        return advantages, returns, (1 - alpha) * outcome_advantages, (1 - alpha) * outcome_returns, alpha * process_advantages, alpha * process_returns
    else:
        advantages = outcome_weight * outcome_advantages + process_weight * process_advantages
        returns = outcome_weight * outcome_returns + process_weight * process_returns
        return advantages, returns, outcome_weight * outcome_advantages, outcome_weight * outcome_returns, process_weight * process_advantages, process_weight * process_returns


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, eos_mask: torch.Tensor,
                                                  gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++. 
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * eos_mask[:, t]

        advantages = verl_F.masked_whiten(returns, eos_mask)
        advantages = advantages * eos_mask

    return advantages, returns

def compute_reinforce_plus_plus_advantage(
    outcome_token_rewards: torch.Tensor,   # (B, L)  – reward on eos_mask
    eos_mask: torch.Tensor,                # (B, L)
    gamma: torch.Tensor,
    process_token_rewards: torch.Tensor | None = None,   # (B, L)
    process_mask: torch.Tensor | None = None,            # (B, L)
    outcome_weight: float = 1.0,
    process_weight: float = 1.0,
    epsilon: float = 1e-6,
    alpha: float | None = None,
):

    outcome_returns = torch.zeros_like(outcome_token_rewards)
    running_return = 0

    for t in reversed(range(outcome_token_rewards.shape[1])):
        running_return = outcome_token_rewards[:, t] + gamma * running_return
        outcome_returns[:, t] = running_return
        running_return = running_return * eos_mask[:, t]

    outcome_advantages = verl_F.masked_whiten(outcome_returns, eos_mask)
    outcome_advantages = outcome_advantages * eos_mask


    if (
        process_token_rewards is None
        or process_mask is None
        or not process_mask.any()
        or process_weight == 0.0
    ):
        return outcome_advantages, outcome_returns
    
    process_returns = torch.zeros_like(process_token_rewards) # (B, n_rollouts, L)

    n_rollouts = process_token_rewards.shape[1]
    for i in range(n_rollouts):
        running_return = 0
        for t in reversed(range(process_token_rewards.shape[1])):
            running_return = process_token_rewards[:, i, t] + gamma * running_return
            process_returns[:, i, t] = running_return
            running_return = running_return * eos_mask[:, t]
    
    process_returns = process_returns.mean(dim=1)    # (B, L)
    process_advantages = verl_F.masked_whiten(process_returns, eos_mask)
    process_advantages = process_advantages * eos_mask

    if alpha is not None:
        advantages = (1 - alpha) * outcome_advantages + alpha * process_advantages
        returns = (1 - alpha) * outcome_returns + alpha * process_returns
        return advantages, returns, (1 - alpha) * outcome_advantages, (1 - alpha) * outcome_returns, alpha * process_advantages, alpha * process_returns
    else:
        advantages = outcome_weight * outcome_advantages + process_weight * process_advantages
        returns = outcome_weight * outcome_returns + process_weight * process_returns
        return advantages, returns, outcome_weight * outcome_advantages, outcome_weight * outcome_returns, process_weight * process_advantages, process_weight * process_returns
    


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor,
                                    eos_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward 
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]

    with torch.no_grad():
        returns = (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange, reweight_coefs=None):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        reweight_coefs: `(torch.Tensor)`
            shape: (bs,)
            The reweight coefficients for the rewards.

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    if reweight_coefs is not None:
        pg_losses = pg_losses * reweight_coefs.unsqueeze(-1)
        pg_losses2 = pg_losses2 * reweight_coefs.unsqueeze(-1)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
