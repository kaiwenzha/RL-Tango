import os
import torch
import uuid
import re
import math
from enum import Enum
import numpy as np
from tqdm import tqdm
from verl import DataProto
from verl.single_controller.ray.base import create_colocated_worker_cls, RayClassWithInitArgs
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role, _timer, compute_advantage, apply_kl_penalty, AdvantageEstimator
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from verl.trainer.ppo import core_algos
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
import pandas as pd
from typing import List
from omegaconf import OmegaConf, open_dict

  
class Mode(int, Enum):
    Generator = 0
    Verifier = 1

def get_alpha(
    step,
    *,                       
    schedule="exp",
    alpha0=0.5,
    alpha_min=1e-3,
    total_steps=1_000_000,   
    exp_target_frac=1.0
):
    if schedule == "none":
        return None

    if step < 0:
        return 0.0
    
    step = np.asarray(step, dtype=float)

    if schedule == "exp":
        k = math.log(alpha0 / alpha_min) / (exp_target_frac * total_steps)
        alpha_val = alpha0 * np.exp(-k * step)
    elif schedule == "linear":
        alpha_val = alpha0 - (alpha0 - alpha_min) * step / total_steps
    elif schedule == "cosine":
        alpha_val = alpha_min + 0.5 * (alpha0 - alpha_min) * (
            1 + np.cos(np.pi * step / total_steps)
        )
    else:
        raise ValueError(f"Unknown schedule: {schedule}. Must be 'exp', 'linear', or 'cosine'")

    return np.maximum(alpha_min, alpha_val)

class RLTangoTrainer(RayPPOTrainer):
    """RLTango Trainer for co-evolving generator and verifier models with RL.
    Both the generator and verifier have their own complete RL training pipelines.
    - Generator is trained with feedback from the verifier (process reward) and the ground truth label (outcome reward).
    - Verifier is trained with its verification correctness labels.
    """
    
    def __init__(self, 
                 config,
                 tokenizer,
                 role_worker_mapping,
                 resource_pool_manager,
                 ray_worker_group_cls,
                 reward_fn_list = List,
                 processor=None,
                 verifier_tokenizer=None,
                 verifier_reward_fn=None):
        """
        Initialize RLTangoTrainer with both generator and verifier components
        
        Args:
            config: Configuration object
            tokenizer: Tokenizer for the generator
            role_worker_mapping: Mapping from Role to Worker classes
            resource_pool_manager: Resource pool manager
            ray_worker_group_cls: Ray worker group class
            reward_fn_list: Reward function for generator
            processor: Processor for multimodal inputs
            verifier_tokenizer: Tokenizer for verifier (can be same as generator)
            verifier_reward_fn: Reward function for verifier (for training verifier)
        """
        super().__init__(config=config,
                         tokenizer=tokenizer,
                         role_worker_mapping=role_worker_mapping,
                         resource_pool_manager=resource_pool_manager,
                         ray_worker_group_cls=ray_worker_group_cls,
                         processor=processor,
                         reward_fn=None, # we will initilize reward_fn later; not using this parent __init__ function
                         val_reward_fn=None) # we will initialize val_reward_fn later; not using this parent __init__ function
        
        self.verifier_tokenizer = verifier_tokenizer
        self.reward_fn_list = reward_fn_list
        
        self.verifier_reward_fn = verifier_reward_fn
        
        # check if we have verifier components
        self.use_verifier = Role.VerifierActorRollout in self.role_worker_mapping
        
        # whether to train the verifier
        self.train_verifier = self.config.verifier.get('trainable', False) if self.use_verifier else False
        
        # setup alternating training mode
        self.alternating_mode =  Mode[self.config.trainer.get("initial_mode", "Generator").capitalize()] if self.train_verifier else Mode.Generator
        
        # verifier critic is used only when adv_estimator is GAE
        if self.train_verifier:
            if self.config.verifier_algorithm.adv_estimator == AdvantageEstimator.GAE:
                self.use_verifier_critic = True
            elif self.config.verifier_algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX, AdvantageEstimator.RLOO
            ]:
                self.use_verifier_critic = False
            else:
                raise NotImplementedError(f"Unknown verifier adv_estimator: {self.config.verifier_algorithm.adv_estimator}")
        else:
            self.use_verifier_critic = False
            
        # verifier reference policy is used when KL penalty is enabled
        self.use_verifier_reference_policy = Role.VerifierRefPolicy in self.role_worker_mapping

        # verifier KL controller
        if self.use_verifier_reference_policy:
            if self.config.verifier_algorithm.kl_ctrl.type == 'fixed':
                self.verifier_kl_ctrl = core_algos.FixedKLController(
                    kl_coef=self.config.verifier_algorithm.kl_ctrl.kl_coef
                )
            elif self.config.verifier_algorithm.kl_ctrl.type == 'adaptive':
                assert config.verifier_algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.verifier_algorithm.kl_ctrl.horizon}'
                self.verifier_kl_ctrl = core_algos.AdaptiveKLController(
                    init_kl_coef=self.config.verifier_algorithm.kl_ctrl.kl_coef,
                    target_kl=self.config.verifier_algorithm.kl_ctrl.target_kl,
                    horizon=self.config.verifier_algorithm.kl_ctrl.horizon
                )
            else:
                raise NotImplementedError
        elif self.train_verifier:
            self.verifier_kl_ctrl = core_algos.FixedKLController(kl_coef=0.)
            
        self.generator_warmup_steps = self.config.trainer.get('generator_warmup_steps', 0)
        self.verifier_warmup_steps = self.config.trainer.get('verifier_warmup_steps', 0)   
        
        # for tracking training phase
        if not self.train_verifier:
            self.training_phase = "generator_only"
        elif self.generator_warmup_steps > 0:
            self.training_phase = "generator_warmup"
        elif self.verifier_warmup_steps > 0:
            self.training_phase = "verifier_warmup"
        else:
            self.training_phase = "alternating"
        
        # calculate training steps for generator and verifier
        self._calculate_training_steps()
        
        self.generator_format_reward_weight = self.config.algorithm.get('generator_format_reward_weight', 1.0)
        
        self.generator_update_step = 0 
        self.verifier_update_step = 0

        # number of verifier inference rollouts (by default, 1)
        self.n_verifier_inference_rollouts = self.config.verifier.rollout.get('n_verifier_inference_rollouts', 1)
        
        # initialize label count tracking for verifier reward reweighting
        self.pos_count_ema = None
        self.neg_count_ema = None
        self.verifier_label_ema_decay = self.config.verifier_algorithm.get("verifier_label_ema_decay", 0.8)
        self.reweight_verifier_rewards = self.config.verifier_algorithm.get("reweight_verifier_rewards", True)
        self.reweight_method = self.config.verifier_algorithm.get("reweight_method", "inverse")

    def _calculate_training_steps(self):
        """
        Calculate the total training steps for generator and verifier based on configuration.
        Sets total_training_steps, gen_total_steps, and ver_total_steps attributes.
        """
        if self.config.trainer.total_training_steps is not None:
            remain_alter_steps = self.config.trainer.total_training_steps - self.generator_warmup_steps - self.verifier_warmup_steps
            assert remain_alter_steps > 0, f"total_training_steps must be larger than generator_warmup_steps + verifier_warmup_steps. Got total_training_steps={self.config.trainer.total_training_steps}, generator_warmup_steps={self.generator_warmup_steps}, verifier_warmup_steps={self.verifier_warmup_steps}"
            
            n_G_steps = self.config.trainer.get("n_generator_steps", -1)
            n_V_steps = self.config.trainer.get("n_verifier_steps", -1)
            
            if not self.train_verifier:
                gen_total_steps = self.generator_warmup_steps + remain_alter_steps
                ver_total_steps = 0
            elif n_G_steps >= 0 and n_V_steps >= 0:
                ratio_g = n_G_steps / (n_G_steps + n_V_steps)
                gen_total_steps = self.generator_warmup_steps + int(remain_alter_steps * ratio_g)
                ver_total_steps = self.verifier_warmup_steps + int(remain_alter_steps * (1 - ratio_g))
            else:
                gen_total_steps = self.generator_warmup_steps + (remain_alter_steps // 2)
                ver_total_steps = self.verifier_warmup_steps + (remain_alter_steps // 2)
            
            self.total_training_steps = self.config.trainer.total_training_steps
            
        else:
            total_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
            remain_alter_steps = total_steps - self.generator_warmup_steps - self.verifier_warmup_steps
            assert remain_alter_steps > 0, f"total_training_steps must be larger than generator_warmup_steps + verifier_warmup_steps. Got total_training_steps={self.config.trainer.total_training_steps}, generator_warmup_steps={self.generator_warmup_steps}, verifier_warmup_steps={self.verifier_warmup_steps}"
            
            n_G_steps = self.config.trainer.get("n_generator_steps", -1)
            n_V_steps = self.config.trainer.get("n_verifier_steps", -1)
            
            if not self.train_verifier:
                gen_total_steps = self.generator_warmup_steps + remain_alter_steps
                ver_total_steps = 0
            elif n_G_steps >= 0 and n_V_steps >= 0:
                ratio_g = n_G_steps / (n_G_steps + n_V_steps)
                gen_total_steps = self.generator_warmup_steps + int(remain_alter_steps * ratio_g)
                ver_total_steps = self.verifier_warmup_steps + int(remain_alter_steps * (1 - ratio_g))
            else:
                gen_total_steps = self.generator_warmup_steps + remain_alter_steps // 2
                ver_total_steps = self.verifier_warmup_steps + remain_alter_steps // 2

            self.total_training_steps = total_steps
        

        self.gen_total_steps = gen_total_steps
        self.ver_total_steps = ver_total_steps
        
        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = gen_total_steps
            self.config.critic.optim.total_training_steps = gen_total_steps
        
            if self.train_verifier:
                self.config.verifier.actor.optim.total_training_steps = ver_total_steps
                self.config.verifier_critic.optim.total_training_steps = ver_total_steps
        
    def init_workers(self):
        """Initialize resource pools and worker groups for both generator and verifier."""

        # initialize resource pools 
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # ------ initialize generator components ------
        
        # create generator actor+rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role='actor_rollout'
            )
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError("Only hybrid engine is supported")

        # create generator critic if needed
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], 
                config=self.config.critic
            )
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create generator reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role='ref'
            )
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create reward model if needed
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model
            )
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # ------ initialize verifier components ------
        
        if self.use_verifier:
            # create verifier actor+rollout
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.VerifierActorRollout)
            verifier_actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.VerifierActorRollout],
                config=self.config.verifier,
                role='actor_rollout'
            )
            self.resource_pool_to_cls[resource_pool]['verifier_actor_rollout'] = verifier_actor_rollout_cls
            
            # create verifier critic if needed
            if self.use_verifier_critic:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.VerifierCritic)
                verifier_critic_cls = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.VerifierCritic], 
                    config=self.config.verifier_critic
                )
                self.resource_pool_to_cls[resource_pool]['verifier_critic'] = verifier_critic_cls
            
            # create verifier reference policy if needed
            if self.use_verifier_reference_policy:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.VerifierRefPolicy)
                verifier_ref_policy_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.VerifierRefPolicy],
                    config=self.config.verifier,
                    role='ref'
                )
                self.resource_pool_to_cls[resource_pool]['verifier_ref'] = verifier_ref_policy_cls

        # initialize worker groups
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)

        # ------ initialize generator worker references ------
        
        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()
            
        # ------ initialize verifier worker references ------
        
        if self.use_verifier:
            
            if self.use_verifier_critic:
                self.verifier_critic_wg = all_wg['verifier_critic']
                self.verifier_critic_wg.init_model()
                
            if self.use_verifier_reference_policy:
                self.verifier_ref_policy_wg = all_wg['verifier_ref']
                self.verifier_ref_policy_wg.init_model()

            self.verifier_actor_rollout_wg = all_wg['verifier_actor_rollout']
            self.verifier_actor_rollout_wg.init_model()
            
        # initialize the generator actor last for better memory estimation
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()
        
    
    def _load_checkpoint(self):
        """Load checkpoints for all models, including verifier components."""
        # load base checkpoints (actor, critic, etc.)
        super()._load_checkpoint()
        
        if self.use_verifier and self.config.trainer.resume_mode != 'disable':
            checkpoint_folder = None
            if self.config.trainer.resume_mode == 'auto':
                checkpoint_folder = self.config.trainer.default_local_dir
                global_step_folder = find_latest_ckpt_path(checkpoint_folder)
                if global_step_folder is None:
                    print('No checkpoint found for verifier')
                    return
            else:
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
            
            # load verifier actor+rollout checkpoint
            verifier_actor_path = os.path.join(global_step_folder, 'verifier_actor')
            if os.path.exists(verifier_actor_path):
                self.verifier_actor_rollout_wg.load_checkpoint(
                    verifier_actor_path,
                    del_local_after_load=self.config.trainer.del_local_ckpt_after_load
                )
                print(f"Loaded verifier actor checkpoint from {verifier_actor_path}")
            else:
                print(f"Warning: No verifier actor checkpoint found at {verifier_actor_path}")
            
            # load verifier critic if needed
            if self.use_verifier_critic:
                verifier_critic_path = os.path.join(global_step_folder, 'verifier_critic')
                if os.path.exists(verifier_critic_path):
                    self.verifier_critic_wg.load_checkpoint(
                        verifier_critic_path,
                        del_local_after_load=self.config.trainer.del_local_ckpt_after_load
                    )
                    print(f"Loaded verifier critic checkpoint from {verifier_critic_path}")
                else:
                    print(f"Warning: No verifier critic checkpoint found at {verifier_critic_path}")
            
            # load verifier reference policy if needed
            if self.use_verifier_reference_policy:
                verifier_ref_path = os.path.join(global_step_folder, 'verifier_ref')
                if os.path.exists(verifier_ref_path):
                    self.verifier_ref_policy_wg.load_checkpoint(
                        verifier_ref_path,
                        del_local_after_load=self.config.trainer.del_local_ckpt_after_load
                    )
                    print(f"Loaded verifier reference policy checkpoint from {verifier_ref_path}")
                else:
                    print(f"Warning: No verifier reference policy checkpoint found at {verifier_ref_path}")
            
        # recover update steps from global steps
        self.recover_update_steps()

    
    def _save_checkpoint(self):
        """Save checkpoints for all models, including verifier components."""
        # save base checkpoints (actor, critic, etc.)
        super()._save_checkpoint()
        
        if self.use_verifier:
            local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                  f'global_step_{self.global_steps}')
            
            # save verifier actor+rollout
            verifier_actor_local_path = os.path.join(local_global_step_folder, 'verifier_actor')
            verifier_actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'verifier_actor')
            
            max_verifier_ckpt_to_keep = self.config.trainer.get('max_verifier_ckpt_to_keep', None)
            
            self.verifier_actor_rollout_wg.save_checkpoint(
                verifier_actor_local_path,
                verifier_actor_remote_path,
                self.global_steps,
                max_ckpt_to_keep=max_verifier_ckpt_to_keep
            )
            
            # save verifier critic if applicable
            if self.use_verifier_critic:
                verifier_critic_local_path = os.path.join(local_global_step_folder, 'verifier_critic')
                verifier_critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                    self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'verifier_critic')
                
                self.verifier_critic_wg.save_checkpoint(
                    verifier_critic_local_path,
                    verifier_critic_remote_path,
                    self.global_steps,
                    max_ckpt_to_keep=max_verifier_ckpt_to_keep
                )

    def _get_verifier_feedback(self, batch: DataProto):
        """Get feedback from the verifier"""
        ground_truth_correctness_reward = self.reward_fn_list[0](batch)
        format_reward = self.reward_fn_list[1](batch)
        
        verification_input, temp_storage = self.verifier_actor_rollout_wg.prepare_verification_input(
            batch, self.tokenizer, n_rollouts=self.n_verifier_inference_rollouts
        )
        batch.meta_info['temp_storage'] = temp_storage
        batch.meta_info['temp_storage']['G_format_reward'] = format_reward.sum(dim=-1) # (B,)
        
        verification_input.meta_info.update({'n': 1})
        verification_output = self.verifier_actor_rollout_wg.generate_sequences(verification_input)
        verification_output.batch['ground_truth_correctness'] = ground_truth_correctness_reward.sum(dim=-1).repeat_interleave(self.n_verifier_inference_rollouts) # (B*n_rollouts,)
        verification_results = self.verifier_actor_rollout_wg.extract_verification_result(
            verification_output, batch, n_rollouts=self.n_verifier_inference_rollouts
        )

        verification_results.batch['G_ground_truth_correctness_reward'] = ground_truth_correctness_reward    # (B, L)
        verification_results.batch['G_format_reward'] = format_reward    # (B, L)

        # extract alignment rates and add to the batch's non_tensor_batch
        if 'alignment_rates' in verification_results.non_tensor_batch:
             batch.meta_info['alignment_rates'] = verification_results.meta_info['alignment_rates']

        # combine the verification results with the original batch
        batch = batch.union(verification_results)

        return batch

    def _train_generator(self, batch: DataProto, metrics, timing_raw):
        """Train the generator"""
        
        # extract input fields for generation
        with _timer('prepare_batch', timing_raw):
            if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                gen_batch = batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                gen_batch = batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )
        
        # generate responses from generator
        with _timer('gen', timing_raw):
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        
        # add generation to original batch
        with _timer('combine_batch', timing_raw):
            batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
            # repeat batch to align with rollout
            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            batch = batch.union(gen_batch_output)
            
            # balance sequence lengths across devices if enabled
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics, logging_prefix='generator/global_seqlen')
            
            # compute batch statistics
            batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
        
        # compute old log probs for generator
        with _timer('old_log_prob', timing_raw):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            batch = batch.union(old_log_prob)
        
        # reference policy log prob if needed for generator
        if self.use_reference_policy:
            with _timer('ref', timing_raw):
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)
        
        # compute values if using critic for generator
        if self.use_critic:
            with _timer('values', timing_raw):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)
        
        # handle reward model if used
        if self.use_rm:
            with _timer('rm_score', timing_raw):
                reward_tensor = self.rm_wg.compute_rm_score(batch)
                rm_scores = reward_tensor.batch['rm_scores']
                metrics['generator/rm_score_mean'] = rm_scores[:,-1].mean().item()
                metrics['generator/rm_score_std'] = rm_scores[:,-1].std().item()
                metrics['generator/rm_score_max'] = rm_scores[:,-1].max().item()
                metrics['generator/rm_score_min'] = rm_scores[:,-1].min().item()
        else:
            rm_scores = 0 
        
        # handle rewards differently depending on training phase
        with _timer('prepare_rewards', timing_raw):
            G_correctness_reward_from_verifier = None
            G_step_rewards = None
            if self.training_phase == "generator_warmup" or (self.training_phase == "generator_only" and not self.use_verifier):
            # during warmup phase, use ground truth rewards directly
                with _timer('calculate_warmup_rewards', timing_raw):
                    # each element in reward_all_list should be (B,L) 
                    G_ground_truth_correctness_reward = self.reward_fn_list[0](batch)
                    G_format_reward = self.reward_fn_list[1](batch)
                    
                    orm_reward_weight =  self.config.reward_model.get('orm_reward_weight', 1.0) if self.use_rm else 0.0
                    G_tmp_reward = (1 - orm_reward_weight) * G_ground_truth_correctness_reward + orm_reward_weight * rm_scores
                    
                    G_final_reward = G_tmp_reward + self.config.algorithm.get("generator_format_reward_weight", 1.0) * G_format_reward
                    batch.batch['token_level_scores'] = G_final_reward

                    # log the mean of number of steps in the generator responses
                    generator_responses = batch.batch['responses']
                    num_steps = []
                    for i in range(generator_responses.shape[0]):
                        r_tokens = generator_responses[i]
                        mask = r_tokens != self.tokenizer.pad_token_id
                        valid_tokens = r_tokens[mask]

                        if len(valid_tokens) > 0:
                            generator_response_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
                            num_steps.append(len(re.findall(r'<step>.*?</step>', generator_response_text, re.DOTALL)))
                        else:
                            num_steps.append(0)
                    metrics['generator/G_step_num_mean'] = np.mean(num_steps)
                    metrics['generator/G_step_num_std'] = np.std(num_steps)
            else:
                # normal training phase with verifier feedback
                with _timer('get_verifier_feedback', timing_raw), torch.no_grad():
                    batch = self._get_verifier_feedback(batch)

                if 'alignment_rates' in batch.meta_info:
                    alignment_rates = batch.meta_info['alignment_rates']
                    metrics['generator/alignment/perfect_alignment_rate'] = alignment_rates.get('perfect_alignment_rate', 0.0)
                    metrics['generator/alignment/missing_steps_rate'] = alignment_rates.get('missing_steps_rate', 0.0)
                    metrics['generator/alignment/extra_steps_rate'] = alignment_rates.get('extra_steps_rate', 0.0)

                ground_truth_weight = self.config.algorithm.get("ground_truth_weight", 0.5)
                mask = (batch.batch['G_final_match_or_not'] == 1).unsqueeze(-1)  # (B, n_rollouts, 1)
                G_ground_truth_correctness_reward = batch.batch['G_ground_truth_correctness_reward'].unsqueeze(1)    # (B, 1, L)
                G_format_reward = batch.batch['G_format_reward']    # (B, L)
                G_correctness_reward_from_verifier = batch.batch['G_correctness_reward_from_verifier']    # (B, n_rollouts, L)

                G_final_reward = torch.where(
                    mask, # (B, n_rollouts, 1)
                    ground_truth_weight * G_ground_truth_correctness_reward + (1.0 - ground_truth_weight) * G_correctness_reward_from_verifier, # (B, n_rollouts, L)
                    G_ground_truth_correctness_reward # (B, 1, L)
                ).mean(dim=1) + self.config.algorithm.get("generator_format_reward_weight", 1.0) * G_format_reward # (B, L)
                batch.batch['token_level_scores'] = G_final_reward

                # process step rewards if available
                if 'G_step_rewards' in batch.batch:
                    G_step_rewards = batch.batch['G_step_rewards']
                else:
                    print(f"No step rewards available for generator training")

            # add additional metrics
            metrics['generator/G_ground_truth_correctness_reward_mean'] = G_ground_truth_correctness_reward.sum(-1).mean().item()
            metrics['generator/G_format_reward_mean'] = G_format_reward.sum(-1).mean().item()
            metrics['generator/G_final_reward_mean'] = G_final_reward.sum(-1).mean().item()
            metrics['generator/G_total_reward_mean'] = batch.batch['token_level_scores'].sum(-1).mean().item()
            
            if G_correctness_reward_from_verifier is not None:
                metrics['generator/G_correctness_reward_from_verifier_mean'] = G_correctness_reward_from_verifier.sum(-1).mean().item()
            if G_step_rewards is not None:
                G_step_rewards_mask = batch.batch['G_step_rewards_mask']
                mask_sum = G_step_rewards_mask.sum(dim=-1)
                per_example_means = torch.where(
                    mask_sum > 0,
                    ((G_step_rewards * (G_step_rewards > 0)) * G_step_rewards_mask).sum(dim=-1), # only sum positive rewards
                    torch.zeros_like(mask_sum)
                )
                metrics['generator/G_step_rewards_mean'] = per_example_means.mean().item()
                metrics['generator/G_step_num_mean'] = mask_sum[mask_sum > 0].float().mean().item()
                metrics['generator/G_step_num_std'] = mask_sum[mask_sum > 0].float().std().item()
        
        # apply KL penalty if needed
        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
            with _timer('apply_kl', timing_raw):
                batch, kl_metrics = apply_kl_penalty(
                    batch,
                    kl_ctrl=self.kl_ctrl,
                    kl_penalty=self.config.algorithm.kl_penalty
                )
                metrics.update(kl_metrics)
        else:
            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
        
        # compute advantages
        with _timer('compute_advantage', timing_raw):
            alpha = get_alpha(
                self.generator_update_step - self.generator_warmup_steps, 
                schedule=self.config.algorithm.alpha_schedule, 
                alpha0=self.config.algorithm.alpha_start, 
                alpha_min=self.config.algorithm.alpha_min,
                total_steps=self.gen_total_steps - self.generator_warmup_steps
            )
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                outcome_weight=self.config.algorithm.get("generator_outcome_reward_weight", 1.0),
                process_weight=self.config.algorithm.get("generator_process_reward_weight", 1.0),
                alpha=alpha
            )
            if alpha is not None:
                metrics['generator/alpha'] = alpha
        
        # update critic if applicable
        if self.use_critic:
            with _timer('update_critic', timing_raw):
                critic_output = self.critic_wg.update_critic(batch)
            metrics.update(reduce_metrics(critic_output.meta_info['metrics'], logging_prefix='generator'))
        
        # only update actor after critic warmup
        if self.config.trainer.critic_warmup <= self.generator_update_step:
            # update actor
            with _timer('update_actor', timing_raw):
                actor_output = self.actor_rollout_wg.update_actor(batch)
            metrics.update(reduce_metrics(actor_output.meta_info['metrics'], logging_prefix='generator'))
        
        # add batch metrics
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, logging_prefix='generator'))

        return batch, metrics
    
    def _train_verifier(self, batch: DataProto, metrics, timing_raw):
        """Train the verifier"""
        
        # extract input fields for generation
        with _timer('prepare_batch', timing_raw):
            if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                gen_batch = batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                gen_batch = batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )
        
        # generate responses from generator
        with _timer('gen', timing_raw), torch.no_grad():
            gen_batch.meta_info = {'n': 1}
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
           
        # add generation to original batch
        with _timer('combine_gen_batch', timing_raw):
            batch = batch.union(gen_batch_output)
        
        # prepare batch for verifier training
        with _timer('prepare_verifier_batch', timing_raw):
            
            ground_truth_correctness = self.reward_fn_list[0](batch).sum(dim=-1)
            G_format_reward = self.reward_fn_list[1](batch).sum(dim=-1)
            
            verifier_batch, temp_storage = self.verifier_actor_rollout_wg.prepare_verification_input(
                batch, self.tokenizer, n_rollouts=1
            )
            verifier_batch.batch['ground_truth_correctness'] = ground_truth_correctness #(B,)
            batch.meta_info['temp_storage'] = temp_storage
            batch.meta_info['temp_storage']['G_format_reward'] = G_format_reward # (B,)
            
        # generate verifier outputs
        with _timer('verifier_generation', timing_raw):
            verifier_gen_batch = verifier_batch.pop(
                batch_keys=['input_ids', 'attention_mask', 'position_ids'],
            )
            verifier_gen_output = self.verifier_actor_rollout_wg.generate_sequences(verifier_gen_batch)
            
        # add verifier's generation to original batch
        with _timer('combine_ver_batch', timing_raw):
            verifier_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(verifier_batch.batch))], dtype=object)
            verifier_batch = verifier_batch.repeat(repeat_times=self.config.verifier.rollout.n, interleave=True)
            verifier_batch = verifier_batch.union(verifier_gen_output)
            batch = batch.repeat(repeat_times=self.config.verifier.rollout.n, interleave=True)
            # repeat interleave for all keys in meta_info['temp_storage'], each value is a list
            for key in batch.meta_info['temp_storage'].keys():
                new_list = []
                for item in batch.meta_info['temp_storage'][key]:
                    new_list.extend([item] * self.config.verifier.rollout.n)
                batch.meta_info['temp_storage'][key] = new_list

            # compute batch statistics
            verifier_batch.meta_info['global_token_num'] = torch.sum(verifier_batch.batch['attention_mask'], dim=-1).tolist()
        
        # extract verification results
        with _timer('extract_verification', timing_raw):
            verification_result = self.verifier_actor_rollout_wg.extract_verification_result(
                verifier_batch, batch, n_rollouts=1
            )

        # compute log probs for verifier outputs
        with _timer('verifier_log_probs', timing_raw):
            log_probs = self.verifier_actor_rollout_wg.compute_log_prob(verifier_batch)
            verifier_batch = verifier_batch.union(log_probs)
        
        # get reference policy log probs if using KL penalty
        if self.use_verifier_reference_policy:
            with _timer('verifier_ref_log_probs', timing_raw):
                ref_log_probs = self.verifier_ref_policy_wg.compute_ref_log_prob(verifier_batch)
                verifier_batch = verifier_batch.union(ref_log_probs)
        
        # compute values if using critic
        if self.use_verifier_critic:
            with _timer('verifier_values', timing_raw):
                values = self.verifier_critic_wg.compute_values(verifier_batch)
                verifier_batch = verifier_batch.union(values)
        
        # calculate reward based on correct verification
        with _timer('verifier_reward_calculation', timing_raw):
            if self.verifier_reward_fn:
                # use custom reward function if provided
                verifier_batch.batch['token_level_scores'] = self.verifier_reward_fn(verifier_batch)
            else:
                V_correctness_reward = verification_result.batch['V_correctness_reward']
                V_format_reward = verification_result.batch['V_format_reward']
                # add classification metrics (balanced accuracy and F1 score)
                if 'verification_label' in verification_result.batch:
                    # extract ground truth and predicted labels
                    ground_truth = (verifier_batch.batch['ground_truth_correctness'] > 0.5).float()
                    predictions = (verification_result.batch['verification_label'] > 0.5).float()
                    
                    # calculate confusion matrix components
                    true_positives = torch.sum((predictions == 1) & (ground_truth == 1)).float()
                    false_positives = torch.sum((predictions == 1) & (ground_truth == 0)).float()
                    false_negatives = torch.sum((predictions == 0) & (ground_truth == 1)).float()
                    true_negatives = torch.sum((predictions == 0) & (ground_truth == 0)).float()
                    
                    # calculate metrics
                    sensitivity = true_positives / (true_positives + false_negatives + 1e-8)
                    specificity = true_negatives / (true_negatives + false_positives + 1e-8)
                    precision = true_positives / (true_positives + false_positives + 1e-8)
                    recall = sensitivity 
                    
                    # balanced accuracy and F1 score
                    balanced_accuracy = (sensitivity + specificity) / 2
                    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
                    
                    # add to metrics
                    metrics['verifier/balanced_accuracy'] = balanced_accuracy.item()
                    metrics['verifier/f1_score'] = f1_score.item()
                    metrics['verifier/precision'] = precision.item()
                    metrics['verifier/recall'] = recall.item()
                    metrics['verifier/accuracy'] = ((true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)).item()
                
                # add reweighting based on class imbalance if enabled
                if self.reweight_verifier_rewards:
                    # V_correctness_reward is (B, L) as one-hot vector
                    # verifier_batch.batch['ground_truth_correctness'] is (B,)
                    assert V_correctness_reward.shape[0] == verifier_batch.batch['ground_truth_correctness'].shape[0], "V_correctness_reward and ground_truth_correctness must have the same batch size"
                    # get counts from current batch
                    correct_examples = (verifier_batch.batch['ground_truth_correctness'] > 0.5).float() # (B,)
                    batch_correct_count = correct_examples.sum().item()
                    batch_incorrect_count = len(correct_examples) - batch_correct_count
                    
                    # initialize EMA counts from first batch or update existing EMAs
                    if self.pos_count_ema is None:
                        self.pos_count_ema = batch_correct_count
                        self.neg_count_ema = batch_incorrect_count
                    else:
                        # update EMA values for counts
                        self.pos_count_ema = self.verifier_label_ema_decay * self.pos_count_ema + (1 - self.verifier_label_ema_decay) * batch_correct_count
                        self.neg_count_ema = self.verifier_label_ema_decay * self.neg_count_ema + (1 - self.verifier_label_ema_decay) * batch_incorrect_count
                    
                    # calculate coefficients based on reweight_method
                    if self.reweight_method == "inverse":
                        pos_coef = 1.0 / (self.pos_count_ema + 1e-6)
                        neg_coef = 1.0 / (self.neg_count_ema + 1e-6)
                    elif self.reweight_method == "sqrt_inverse":
                        pos_coef = 1.0 / (math.sqrt(self.pos_count_ema) + 1e-6)
                        neg_coef = 1.0 / (math.sqrt(self.neg_count_ema) + 1e-6)
                    else:
                        raise ValueError(f"Unknown reweight_method: {self.reweight_method}. Must be 'inverse' or 'sqrt_inverse'")
                    
                    # create per-example coefficients
                    reweight_coefs = torch.where(
                        correct_examples > 0.5, 
                        pos_coef * torch.ones_like(correct_examples),
                        neg_coef * torch.ones_like(correct_examples)
                    )    # (B,)
                    scale_factor = len(reweight_coefs) / torch.sum(reweight_coefs)
                    reweight_coefs = reweight_coefs * scale_factor
                    
                    verifier_batch.batch['reweight_coefs'] = reweight_coefs
                    
                    # log metrics
                    metrics['verifier/pos_count'] = batch_correct_count
                    metrics['verifier/neg_count'] = batch_incorrect_count
                    metrics['verifier/pos_count_ema'] = self.pos_count_ema
                    metrics['verifier/neg_count_ema'] = self.neg_count_ema
                    metrics['verifier/pos_coef'] = scale_factor * pos_coef
                    metrics['verifier/neg_coef'] = scale_factor * neg_coef
                    metrics['verifier/scale_fator'] = scale_factor
                
                verifier_batch.batch['token_level_scores'] = V_correctness_reward * self.config.verifier_algorithm.get("verifier_correctness_reward_weight", 1.0) + V_format_reward * self.config.verifier_algorithm.get("verifier_format_reward_weight", 1.0)
                
                # track metrics
                metrics['verifier/V_correctness_reward_mean'] = V_correctness_reward.sum(-1).mean().item()
                metrics['verifier/V_format_reward_mean'] = V_format_reward.sum(-1).mean().item()
                metrics['verifier/V_final_reward_mean'] = verifier_batch.batch['token_level_scores'].sum(-1).mean().item()
                
                # also track step-wise metrics if available (only for logging, not used for training)
                if 'G_step_rewards' in verification_result.batch:
                    G_step_rewards = verification_result.batch['G_step_rewards']    # (B, n_rollouts, L)
                    G_step_rewards_mask = verification_result.batch['G_step_rewards_mask']   # (B, n_rollouts, L)
                    mask_sum = G_step_rewards_mask.sum(dim=-1)
                    per_example_means = torch.where(
                        mask_sum > 0,
                        ((G_step_rewards * (G_step_rewards > 0)) * G_step_rewards_mask).sum(dim=-1), # only sum positive rewards
                        torch.zeros_like(mask_sum)
                    )
                    metrics['verifier/G_step_rewards_mean'] = per_example_means.mean().item()
                    metrics['verifier/G_step_num_mean'] = mask_sum[mask_sum > 0].float().mean().item()
                    metrics['verifier/G_step_num_std'] = mask_sum[mask_sum > 0].float().std().item()
        
        # apply KL penalty if using reference policy
        if not self.config.verifier.actor.get('use_kl_loss', False):
            with _timer('verifier_kl', timing_raw):
                verifier_batch, kl_metrics = apply_kl_penalty(
                    verifier_batch,
                    kl_ctrl=self.verifier_kl_ctrl,
                    kl_penalty=self.config.verifier_algorithm.kl_penalty
                )
                verifier_kl_metrics = {f'verifier/{k}': v for k, v in kl_metrics.items()}
                metrics.update(verifier_kl_metrics)
        else:
            verifier_batch.batch['token_level_rewards'] = verifier_batch.batch['token_level_scores']
        
        
        # compute advantages
        with _timer('verifier_advantage', timing_raw):
            verifier_batch = compute_advantage(
                verifier_batch,
                adv_estimator=self.config.verifier_algorithm.adv_estimator,
                gamma=self.config.verifier_algorithm.gamma,
                lam=self.config.verifier_algorithm.lam,
            )
        
        # update verifier critic if applicable
        if self.use_verifier_critic:
            with _timer('update_verifier_critic', timing_raw):
                critic_output = self.verifier_critic_wg.update_critic(verifier_batch)
            metrics.update(reduce_metrics(critic_output.meta_info['metrics'], logging_prefix='verifier'))
        
        # only update verifier actor after critic warmup
        if self.config.trainer.verifier_critic_warmup <= self.verifier_update_step:
            # update verifier actor
            with _timer('update_verifier_actor', timing_raw):
                actor_output = self.verifier_actor_rollout_wg.update_actor(verifier_batch)
            metrics.update(reduce_metrics(actor_output.meta_info['metrics'], logging_prefix='verifier'))
        
        # add batch metrics
        metrics.update(compute_data_metrics(batch=verifier_batch, use_critic=self.use_verifier_critic, logging_prefix = 'verifier'))
        
        return verifier_batch, metrics
    
    def _calculate_data_source_metrics(
        self,
        generator_total_scores,
        generator_correctness_scores,
        generator_format_scores,
        data_source_lst,
        verification_labels,
        verifier_total_scores,
        verifier_correctness_scores,
        verifier_format_scores,
        step_rewards_lst,
    ):
        """Calculate metrics for both generator and verifier, broken down by data source."""
        generator_metrics = {}
        verifier_metrics = {}
        
        np_generator_total_scores = np.array(generator_total_scores)
        np_generator_correctness_scores = np.array(generator_correctness_scores)
        np_generator_format_scores = np.array(generator_format_scores)
        
        all_data_sources = np.concatenate(data_source_lst, axis=0)
        unique_data_sources = set(all_data_sources)
        
        # overall generator metrics
        generator_metrics['val/generator/total_score/all'] = np.mean(np_generator_total_scores)
        generator_metrics['val/generator/correctness_score/all'] = np.mean(np_generator_correctness_scores)
        generator_metrics['val/generator/format_score/all'] = np.mean(np_generator_format_scores)
        
        # add good format ratio for generator (all data sources)
        good_format_count = sum(1 for r in generator_format_scores if r > 0.8)
        generator_metrics['val/generator/good_format_ratio/all'] = good_format_count / len(generator_format_scores) if generator_format_scores else 0
        
        # per-data-source generator metrics
        for data_source in unique_data_sources:
            source_mask = all_data_sources == data_source
            
            if not np.any(source_mask):
                continue
            
            generator_metrics[f'val/generator/total_score/{data_source}'] = np.mean(np_generator_total_scores[source_mask])
            generator_metrics[f'val/generator/correctness_score/{data_source}'] = np.mean(np_generator_correctness_scores[source_mask])
            generator_metrics[f'val/generator/format_score/{data_source}'] = np.mean(np_generator_format_scores[source_mask])
            
            ds_format_scores = np_generator_format_scores[source_mask]
            good_format_count_ds = sum(1 for r in ds_format_scores if r > 0.8)
            generator_metrics[f'val/generator/good_format_ratio/{data_source}'] = good_format_count_ds / len(ds_format_scores) if len(ds_format_scores) > 0 else 0
            
        # verifier metrics
        if self.use_verifier and verification_labels:
            np_verifier_total_scores = np.array(verifier_total_scores)
            np_verifier_correctness_scores = np.array(verifier_correctness_scores)
            np_verifier_format_scores = np.array(verifier_format_scores)
            
            # add overall score metrics
            verifier_metrics['val/verifier/correctness_score/all'] = np.mean(np_verifier_correctness_scores)
            verifier_metrics['val/verifier/format_score/all'] = np.mean(np_verifier_format_scores)
            verifier_metrics['val/verifier/total_score/all'] = np.mean(np_verifier_total_scores)
            
            # calculate per-data-source metrics
            for data_source in unique_data_sources:
                source_mask = all_data_sources == data_source
                
                if not np.any(source_mask):
                    continue
                
                verifier_metrics[f'val/verifier/correctness_score/{data_source}'] = np.mean(np_verifier_correctness_scores[source_mask])
                verifier_metrics[f'val/verifier/format_score/{data_source}'] = np.mean(np_verifier_format_scores[source_mask])
                verifier_metrics[f'val/verifier/total_score/{data_source}'] = np.mean(np_verifier_total_scores[source_mask])
            
            # add step rewards metrics if available
            if step_rewards_lst:
                step_rewards_np = np.array(step_rewards_lst)
                
                verifier_metrics['val/verifier/step_rewards_mean/all'] = np.mean(step_rewards_np)
                verifier_metrics['val/verifier/step_rewards_std/all'] = np.std(step_rewards_np)
                
                # per-data-source step rewards
                for data_source in unique_data_sources:
                    source_mask = all_data_sources == data_source
                    
                    if np.any(source_mask):
                        verifier_metrics[f'val/verifier/step_rewards_mean/{data_source}'] = np.mean(step_rewards_np[source_mask])
                        verifier_metrics[f'val/verifier/step_rewards_std/{data_source}'] = np.std(step_rewards_np[source_mask])
            
            # format quality metric
            good_format_count = sum(1 for r in verifier_format_scores if r > 0.8)
            verifier_metrics['val/verifier/good_format_ratio/all'] = good_format_count / len(verifier_format_scores)
            
            # add good format ratio per data source for verifier
            for data_source in unique_data_sources:
                source_mask = all_data_sources == data_source
                
                if np.any(source_mask):
                    v_ds_format_scores = np_verifier_format_scores[source_mask]
                    v_good_format_count_ds = sum(1 for r in v_ds_format_scores if r > 0.8)
                    verifier_metrics[f'val/verifier/good_format_ratio/{data_source}'] = v_good_format_count_ds / len(v_ds_format_scores) if len(v_ds_format_scores) > 0 else 0
        
        return {**generator_metrics, **verifier_metrics}

    def _validate(self):
        """Validate both generator and verifier models."""
        
        # for logging examples; these variables will be called with .extend(), so won't be nested if len(self.val_dataloader) > 1
        generator_total_scores = []
        generator_correctness_scores = []
        generator_format_scores = []
        generator_prompts = []
        generator_responses = []
        ground_truth_solutions = []

        verifier_total_scores = []
        verifier_correctness_scores = []
        verifier_format_scores = []
        verifier_prompts = []
        verifier_responses = []

        verification_labels = []
        
        # for aggregating metrics; these variables will be called with .append(); so will be nested if len(self.val_dataloader) > 1
        data_source_lst = []
        step_rewards_lst = []

        val_inference_rollouts = 1
        
        # process each validation batch
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            
            batch_gt_solutions = [item['ground_truth'] for item in test_batch.non_tensor_batch['reward_model']]
            batch_gt_solutions = [gt for gt in batch_gt_solutions for _ in range(self.config.actor_rollout_ref.rollout.val_kwargs.n)]
            ground_truth_solutions.extend(batch_gt_solutions)
            
            # repeat test batch to match the behavior in ray_trainer.py
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)
            
            # store original problems
            input_ids = test_batch.batch['input_ids']
            problem_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            generator_prompts.extend(problem_texts)
            
            # extract input fields for generation
            if 'multi_modal_inputs' in test_batch.non_tensor_batch:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )
            
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            
            # generate responses from generator
            test_gen_batch_padded, gen_pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size)
            
            with torch.no_grad():
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            
            # unpad the generator output
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=gen_pad_size)
            
            # store generator solutions
            solution_ids = test_output_gen_batch.batch['responses']
            solution_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in solution_ids]
            generator_responses.extend(solution_texts)
            
            # combine with original batch
            test_batch = test_batch.union(test_output_gen_batch)

            ground_truth_correctness = self.reward_fn_list[0](test_batch).sum(dim=-1)  # (B,)
            generator_format_score = self.reward_fn_list[1](test_batch).sum(dim=-1)  # (B,)
            
            generator_total_score = ground_truth_correctness + self.config.algorithm.get("generator_format_reward_weight", 1.0) * generator_format_score
            generator_total_scores.extend(generator_total_score.cpu().tolist())
            generator_correctness_scores.extend(ground_truth_correctness.cpu().tolist())
            generator_format_scores.extend(generator_format_score.cpu().tolist())

            # store reward tensors for later metric calculation
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * generator_total_score.shape[0]))
            
            if not self.use_verifier:
                continue
            
            # pad the batch for verifier
            test_ver_batch_padded, ver_pad_size = pad_dataproto_to_divisor(
                test_batch, self.verifier_actor_rollout_wg.world_size)
            
            # prepare batch for verifier
            verifier_input_batch, temp_storage = self.verifier_actor_rollout_wg.prepare_verification_input(
                test_ver_batch_padded, self.tokenizer, n_rollouts=val_inference_rollouts
            )
            test_ver_batch_padded.meta_info['temp_storage'] = temp_storage
            test_ver_batch_padded.meta_info['temp_storage']['G_format_reward'] = torch.cat([generator_format_score, generator_format_score[:ver_pad_size]], dim = 0) # (B,)

            if ver_pad_size == 0:
                verifier_prompts.extend([self.verifier_tokenizer.decode(ids, skip_special_tokens=True) for i, ids in enumerate(verifier_input_batch.batch['input_ids']) if i % self.config.actor_rollout_ref.rollout.val_kwargs.n == 0])
            elif ver_pad_size > 0:
                verifier_prompts.extend([self.verifier_tokenizer.decode(ids, skip_special_tokens=True) for i, ids in enumerate(verifier_input_batch.batch['input_ids'][:-ver_pad_size]) if i % self.config.actor_rollout_ref.rollout.val_kwargs.n == 0])

            verifier_input_batch.meta_info.update({
                'eos_token_id': self.verifier_tokenizer.eos_token_id,
                'pad_token_id': self.verifier_tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.verifier.rollout.val_kwargs.do_sample,
                'validate': True,
            })
            
            # generate verification
            with torch.no_grad():
                verifier_output_batch = self.verifier_actor_rollout_wg.generate_sequences(verifier_input_batch)
            
            verifier_output_batch.batch['ground_truth_correctness'] = torch.cat(
                [ground_truth_correctness, ground_truth_correctness[:ver_pad_size]], dim=0
            ).repeat_interleave(val_inference_rollouts) # (B*n_rollouts,)
            verification_results = self.verifier_actor_rollout_wg.extract_verification_result(
                verifier_output_batch, test_ver_batch_padded, n_rollouts=val_inference_rollouts
            )
            verification_results = unpad_dataproto(verification_results, pad_size=ver_pad_size)
            
            for key, value in verification_results.meta_info.items():
                print(f"key = {key}, value = {value}, len(value) = {len(value) if hasattr(value, '__len__') else 'N/A'}")
                
            # extract verifier judgments
            verification_label = verification_results.batch['verification_label']
            verifier_correctness_score = verification_results.batch['V_correctness_reward'].sum(-1)
            verifier_format_score = verification_results.batch['V_format_reward'].sum(-1)
            verifier_total_score = verifier_correctness_score + self.config.verifier_algorithm.get("verifier_format_reward_weight", 1.0) * verifier_format_score

            verification_labels.extend(verification_label.cpu().tolist())
            verifier_correctness_scores.extend(verifier_correctness_score.cpu().tolist())
            verifier_format_scores.extend(verifier_format_score.cpu().tolist())
            verifier_total_scores.extend(verifier_total_score.cpu().tolist())
            verifier_responses.extend(verification_results.non_tensor_batch['verifier_responses'])
            
            # collect step rewards
            G_step_rewards = verification_results.batch['G_step_rewards']   # (B, n_rollouts, L)
            G_step_rewards_mask = verification_results.batch['G_step_rewards_mask']   # (B, n_rollouts, L)
            mask_sum = G_step_rewards_mask.sum(dim=-1)  # (B, n_rollouts)
            per_example_means = torch.where(
                mask_sum > 0,
                ((G_step_rewards * (G_step_rewards > 0)) * G_step_rewards_mask).sum(dim=-1), # only sum positive rewards
                torch.zeros_like(mask_sum)
            ).mean(dim=-1)  # (B,)
            step_rewards_lst.extend(per_example_means.cpu().tolist())
        
            
        combined_metrics = self._calculate_data_source_metrics(
            generator_total_scores,
            generator_correctness_scores,
            generator_format_scores,
            data_source_lst,
            verification_labels,
            verifier_total_scores,
            verifier_correctness_scores,
            verifier_format_scores,
            step_rewards_lst,
        )

        # log generator examples
        self._maybe_log_val_generations(
            generator_prompts, 
            generator_responses, 
            ground_truth_solutions,
            generator_total_scores,
            generator_correctness_scores,
            generator_format_scores,
        )

        # log verification examples if verifier is used
        if self.use_verifier:
            self._maybe_log_val_verifications(
                generator_prompts, 
                generator_responses, 
                ground_truth_solutions,
                generator_correctness_scores, 
                verification_labels, 
                verifier_responses,
                verifier_prompts,
                verifier_correctness_scores,
                verifier_format_scores,
                step_rewards_lst
            )

       
        return combined_metrics
    
    def fit(self):
        """Training loop that follows a three-phase schedule:
        1. Generator warmup: train only generator for generator_warmup_steps (if generator_warmup_steps > 0)
        2. Verifier warmup: train only verifier for verifier_warmup_steps (if verifier_warmup_steps > 0)
        3. Alternating: train generator and verifier alternately
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True)
        )

        for backend in self.config.trainer.logger:
            if backend == 'wandb':
                import wandb
                wandb.define_metric("generator/*", step_metric="generator_step")
                wandb.define_metric("verifier/*", step_metric="verifier_step")
                wandb.define_metric("val/generator/*", step_metric="generator_step")
                wandb.define_metric("val/verifier/*", step_metric="verifier_step")
            else:
                raise NotImplementedError(f"Please implement separate step logging for verifier and generator: {backend}")

        self.global_steps = 0

        self._load_checkpoint()

        # perform validation before training
        if self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            val_metrics.update({'generator_step': self.generator_update_step, 'verifier_step': self.verifier_update_step})
            logger.log(val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None

        # determine alternating frequency for alternating phase
        n_G_steps = self.config.trainer.get("n_generator_steps", -1)
        n_V_steps = self.config.trainer.get("n_verifier_steps", -1)
        if n_G_steps >= 0 and n_V_steps >= 0:
            use_step_counters = True
            # initialize to 0 or use recovered values if available
            consecutive_g_steps = getattr(self, 'consecutive_g_steps', 0)
            consecutive_v_steps = getattr(self, 'consecutive_v_steps', 0)
            # clean up attributes if they were temporarily set
            if hasattr(self, 'consecutive_g_steps'):
                delattr(self, 'consecutive_g_steps')
            if hasattr(self, 'consecutive_v_steps'):
                delattr(self, 'consecutive_v_steps')
        else:
            alt_freq = self.config.trainer.get("alternating_frequency", 1) if self.train_verifier else 1e9
            use_step_counters = False
        
        # track when we should switch phases
        generator_warmup_end = self.generator_warmup_steps
        verifier_warmup_end = generator_warmup_end + self.verifier_warmup_steps

        # set appropriate initial mode if one mode has zero steps
        if n_G_steps == 0 and n_V_steps > 0:
            # if generator steps is zero, force verifier mode
            self.alternating_mode = Mode.Verifier
        elif n_V_steps == 0 and n_G_steps > 0:
            # if verifier steps is zero, force generator mode
            self.alternating_mode = Mode.Generator

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                
                # convert to DataProto
                original_batch = DataProto.from_single_dict(batch_dict)
                
                # determine training mode based on current step
                if not self.train_verifier:
                    self.training_phase = "generator_only"
                    current_mode = Mode.Generator
                elif self.global_steps <= generator_warmup_end:
                    # phase 1: generator warmup
                    self.training_phase = "generator_warmup"
                    current_mode = Mode.Generator
                elif self.global_steps <= verifier_warmup_end:
                    # phase 2: verifier warmup
                    self.training_phase = "verifier_warmup"
                    current_mode = Mode.Verifier
                else:
                    # phase 3: alternating
                    self.training_phase = "alternating"
                    # use the current alternating_mode which switches every alt_freq steps
                    current_mode = self.alternating_mode
                
                # update progress bar with phase and mode information
                if self.training_phase == "alternating":
                    if use_step_counters:
                        if current_mode == Mode.Generator:
                            progress_bar.set_description(f"Training [Phase: {self.training_phase}, Mode: {Mode(current_mode).name}, G Steps: {consecutive_g_steps + 1}/{n_G_steps}, V Steps: {n_V_steps}]")
                        else:
                            progress_bar.set_description(f"Training [Phase: {self.training_phase}, Mode: {Mode(current_mode).name}, G Steps: {n_G_steps}, V Steps: {consecutive_v_steps + 1}/{n_V_steps}]")
                    else:
                        progress_bar.set_description(f"Training [Phase: {self.training_phase}, Mode: {Mode(current_mode).name}, Alt Freq: {alt_freq}]")
                else:
                    progress_bar.set_description(f"Training [Phase: {self.training_phase}, Mode: {Mode(current_mode).name}]")
                    
                with _timer('step', timing_raw):
                    if current_mode == Mode.Generator:
                        batch, metrics = self._train_generator(original_batch, metrics, timing_raw)
                        self.generator_update_step += 1
                    elif current_mode == Mode.Verifier:
                        batch, metrics = self._train_verifier(original_batch, metrics, timing_raw)
                        self.verifier_update_step += 1
                    else:
                        raise ValueError(f"Invalid mode: {current_mode}")
                
                # only switch modes in the alternating phase
                if self.training_phase == "alternating":
                    if use_step_counters:
                        # using separate counters for G and V steps
                        if current_mode == Mode.Generator:
                            consecutive_g_steps += 1
                            if consecutive_g_steps >= n_G_steps:
                                if n_V_steps > 0:
                                    self.alternating_mode = Mode.Verifier
                                consecutive_g_steps = 0
                        else:  # Mode.Verifier
                            consecutive_v_steps += 1
                            if consecutive_v_steps >= n_V_steps:
                                if n_G_steps > 0:
                                    self.alternating_mode = Mode.Generator
                                consecutive_v_steps = 0
                    elif self.global_steps % alt_freq == 0:
                        # legacy mode with single alt_freq
                        self.alternating_mode = Mode((self.alternating_mode + 1) % len(Mode))
                
                # add the current phase to metrics
                metrics['training_phase'] = self.training_phase
                
                # run validation if needed
                is_last_step = self.global_steps >= self.total_training_steps
                is_ver_warmup_end = self.global_steps == verifier_warmup_end
                if self.config.trainer.test_freq > 0 and (is_last_step or (is_ver_warmup_end and (self.generator_warmup_steps > 0 or self.verifier_warmup_steps > 0)) \
                        or (self.generator_update_step % self.config.trainer.test_freq == 0 and current_mode == Mode.Generator)):
                    with _timer('validation', timing_raw):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
                
                # save checkpoint if needed
                if self.config.trainer.save_freq > 0 and (is_last_step or (is_ver_warmup_end and (self.generator_warmup_steps > 0 or self.verifier_warmup_steps > 0)) \
                        or (self.generator_update_step % self.config.trainer.save_freq == 0 and current_mode == Mode.Generator) or self.global_steps == 1):
                    with _timer('save_checkpoint', timing_raw):
                        self._save_checkpoint()
            
                # update metrics
                metrics.update(compute_timing_metrics(batch = batch, timing_raw=timing_raw))
                
                # compute throughput metrics
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch = batch, timing_raw=timing_raw, n_gpus=n_gpus))
                
                metrics.update({
                    'generator_step': self.generator_update_step,
                    'verifier_step': self.verifier_update_step,
                })
                logger.log(metrics, step=self.global_steps)
                    
                if is_last_step:
                    print(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    return
                
                progress_bar.update(1)
                self.global_steps += 1

    def _maybe_log_val_generations(self, problems, solutions, ground_truth_solutions, total_scores, correctness_scores, format_scores):
        """Log a table of generator examples to the configured logger with generation metrics."""
        generations_to_log = self.config.trainer.get('val_generations_to_log_to_wandb', 10)
        if generations_to_log == 0:
            return
        
        import numpy as np
        
        # create tuples with all available metrics
        samples = list(zip(problems, solutions, ground_truth_solutions, total_scores, correctness_scores, format_scores))
        
        # use fixed random seed for deterministic selection
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        
        # take first N samples after shuffling
        samples = samples[:generations_to_log]
        
        # format for logging
        generation_examples = []
        for problem, solution, ground_truth_solution, total, correctness, format_score in samples:
            is_correct = "✓" if correctness > 0.5 else "✗"
            formatted_example = {
                'Global Step': self.global_steps,
                "Generator Step": self.generator_update_step,
                "Verifier Step": self.verifier_update_step,
                "Problem": problem,
                "Solution": solution,
                "Ground-truth Solution": ground_truth_solution,
                "Correct": is_correct,
                "Total Score": f"{total:.3f}",
                "Correctness": f"{correctness:.3f}",
                "Format": f"{format_score:.3f}"
            }
            generation_examples.append(formatted_example)
        
        # log to each configured logger
        for backend in self.config.trainer.logger:
            if backend == 'wandb':
                import wandb
                wandb.log({
                    "generator_examples": wandb.Table(dataframe=pd.DataFrame(generation_examples))
                }, step=self.global_steps)
            else:
                raise NotImplementedError(f"Logging for {backend} is not implemented.")

    def _maybe_log_val_verifications(self, problems, solutions, ground_truth_solutions, ground_truth_correctnesses, 
                                    verification_labels, verifier_responses, 
                                    verifier_prompts, verifier_correctness_scores, verifier_format_scores, step_rewards_lst):
        """Log a table of verification examples to the configured logger with verification metrics."""
        verifications_to_log = self.config.trainer.get('val_verifications_to_log_to_wandb', 10)
        if verifications_to_log == 0:
            return
        
        import numpy as np
        
        # create tuples with all available metrics
        samples = list(zip(problems, solutions, ground_truth_solutions, ground_truth_correctnesses, 
                        verification_labels, verifier_responses,
                        verifier_prompts, verifier_correctness_scores, verifier_format_scores, step_rewards_lst))
        
        # use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        
        # take first N samples after shuffling
        samples = samples[:verifications_to_log]
        
        # format samples for logging
        verification_examples = []
        
        # table with detailed metrics
        for item in samples:
            problem, solution, ground_truth_solution, ground_truth_correctness, verifier_judgment, verifier_response, verifier_prompt, v_correctness, v_format, step_rewards = item
            
            # determine labels and symbols
            gt_label = "Correct" if ground_truth_correctness > 0.5 else "Incorrect"
            verifier_label = "Correct" if verifier_judgment > 0.5 else "Incorrect"
            verifier_accuracy = "✓" if (ground_truth_correctness > 0.5) == (verifier_judgment > 0.5) else "✗"
            
            # create a detailed row
            verification_examples.append({
                "Global Step": self.global_steps,
                "Generator Step": self.generator_update_step,
                "Verifier Step": self.verifier_update_step,
                "Problem": problem,
                "Solution": solution,
                "Ground-truth Solution": ground_truth_solution,
                "Verifier Prompt": verifier_prompt,
                "Verifier Response": verifier_response,
                "Ground-truth Correctness": gt_label,
                "Verifier Judgment": verifier_label,
                "Verifier Accuracy": verifier_accuracy,
                "Verifier Correctness Score": f"{v_correctness:.3f}",
                "Verifier Format Score": f"{v_format:.3f}",
                "Step Score": f"{step_rewards:.3f}",
            })
        
        # log to each configured logger
        from verl.utils.tracking import Tracking
        for backend in self.config.trainer.logger:
            if backend == 'wandb':
                import wandb
                wandb.log({"verification_examples": wandb.Table(dataframe=pd.DataFrame(verification_examples))}, 
                         step=self.global_steps)
            else:
                raise NotImplementedError(f"Logging for {backend} is not implemented.")

    def recover_update_steps(self):
        """Recover update steps and alternating state from global_steps."""
        if not self.train_verifier:
            # if only training generator
            self.generator_update_step = self.global_steps
            self.verifier_update_step = 0
            return
        
        # initialize these variables that will be used if in alternating phase
        consecutive_g_steps = 0
        consecutive_v_steps = 0
        
        # determine where we are in the training schedule
        if self.global_steps <= self.generator_warmup_steps:
            # still in generator warmup phase
            self.generator_update_step = self.global_steps
            self.verifier_update_step = 0
        elif self.global_steps <= (self.generator_warmup_steps + self.verifier_warmup_steps):
            # in verifier warmup phase
            self.generator_update_step = self.generator_warmup_steps
            self.verifier_update_step = self.global_steps - self.generator_warmup_steps
        else:
            # in alternating phase
            remain_steps = self.global_steps - self.generator_warmup_steps - self.verifier_warmup_steps
            
            # handle based on step counting method
            n_G_steps = self.config.trainer.get("n_generator_steps", -1)
            n_V_steps = self.config.trainer.get("n_verifier_steps", -1)
            
            if n_G_steps >= 0 and n_V_steps >= 0:
                # using step counters with specific G:V ratio
                cycle_length = n_G_steps + n_V_steps
                cycles_completed = remain_steps // cycle_length
                steps_in_current_cycle = remain_steps % cycle_length
                
                # full G and V steps from completed cycles
                g_steps_in_completed_cycles = cycles_completed * n_G_steps
                v_steps_in_completed_cycles = cycles_completed * n_V_steps
                
                # determine mode and consecutive steps in current cycle
                if n_G_steps == 0:
                    # generator steps is zero, always use Verifier mode
                    self.alternating_mode = Mode.Verifier
                    consecutive_g_steps = 0
                    consecutive_v_steps = steps_in_current_cycle
                    
                    # total steps
                    self.generator_update_step = self.generator_warmup_steps
                    self.verifier_update_step = self.verifier_warmup_steps + v_steps_in_completed_cycles + consecutive_v_steps
                elif n_V_steps == 0:
                    # verifier steps is zero, always use Generator mode
                    self.alternating_mode = Mode.Generator
                    consecutive_g_steps = steps_in_current_cycle
                    consecutive_v_steps = 0
                    
                    # total steps
                    self.generator_update_step = self.generator_warmup_steps + g_steps_in_completed_cycles + consecutive_g_steps
                    self.verifier_update_step = self.verifier_warmup_steps
                elif steps_in_current_cycle < n_G_steps:
                    self.alternating_mode = Mode.Generator
                    consecutive_g_steps = steps_in_current_cycle
                    consecutive_v_steps = 0
                    
                    # total steps
                    self.generator_update_step = self.generator_warmup_steps + g_steps_in_completed_cycles + consecutive_g_steps
                    self.verifier_update_step = self.verifier_warmup_steps + v_steps_in_completed_cycles
                else:
                    # currently in Verifier mode
                    self.alternating_mode = Mode.Verifier
                    consecutive_g_steps = 0
                    consecutive_v_steps = steps_in_current_cycle - n_G_steps
                    
                    # total steps
                    self.generator_update_step = self.generator_warmup_steps + g_steps_in_completed_cycles + n_G_steps
                    self.verifier_update_step = self.verifier_warmup_steps + v_steps_in_completed_cycles + consecutive_v_steps
                
                self.consecutive_g_steps = consecutive_g_steps
                self.consecutive_v_steps = consecutive_v_steps
            else:
                # using fixed alternating frequency alt_freq=1
                if remain_steps % 2 == 0:
                    self.alternating_mode = Mode.Generator if self.config.trainer.get("initial_mode", "Generator").capitalize() == "Generator" else Mode.Verifier
                else:
                    self.alternating_mode = Mode.Verifier if self.config.trainer.get("initial_mode", "Generator").capitalize() == "Generator" else Mode.Generator
                    
                # calculate update steps assuming equal division
                self.generator_update_step = self.generator_warmup_steps + (remain_steps + 1) // 2
                self.verifier_update_step = self.verifier_warmup_steps + remain_steps // 2