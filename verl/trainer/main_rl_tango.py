"""Main script for RL Tango training."""
import os
import ray
import hydra
from verl.trainer.main_ppo import get_custom_reward_fn
from verl.utils.reward_score.format import format_compute_score
from verl.trainer.ppo.rl_tango_trainer import RLTangoTrainer

@hydra.main(config_path='config', config_name='rl_tango_trainer', version_base=None)
def main(config):
    run_rltango(config)

def run_rltango(config) -> None:

    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not ray.is_initialized():
        # initialize ray cluster
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN'
            }
        },
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

@ray.remote(num_cpus=1)  
class TaskRunner:
    def run(self, config):
        from verl.utils.fs import copy_to_local
        from pprint import pprint
        from omegaconf import OmegaConf
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values

        OmegaConf.resolve(config)

        gen_local_path = copy_to_local(config.actor_rollout_ref.model.path)
        
        # instantiate generator tokenizer
        from verl.utils import hf_tokenizer, hf_processor
    
        gen_tokenizer = hf_tokenizer(gen_local_path)
        
        processor = hf_processor(gen_local_path, use_fast=True)  # used for multimodal LLM, could be none
        
        # download and initialize verifier tokenizer if enabled
        verifier_tokenizer = None
        if config.verifier.enable:
            verifier_local_path = copy_to_local(config.verifier.model.path)
            
            # use separate tokenizer if verifier model is different from generator
            if config.verifier.model.path != config.actor_rollout_ref.model.path:
                verifier_tokenizer = hf_tokenizer(verifier_local_path)
            else:
                verifier_tokenizer = gen_tokenizer

        # define worker classes based on strategy
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == 'megatron':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role


        # initialize role to worker class mapping for generator components
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
        }

        # add verifier workers if enabled
        if config.verifier.enable:
            # import verifier worker classes based on strategy
            if config.verifier.actor.strategy == 'fsdp':
                assert config.verifier.actor.strategy == config.verifier_critic.strategy
                from verl.workers.verifier_worker import VerifierActorRolloutWorker, VerifierCriticWorker
            else:
                raise NotImplementedError(f"verifier worker not implemented for the config.verifier.actor.strategy = {config.verifier.actor.strategy}")
            
            # add verifier actor+rollout
            role_worker_mapping[Role.VerifierActorRollout] = ray.remote(VerifierActorRolloutWorker)

            
            # add verifier training components if trainable
            if config.verifier.get('trainable', False):
                role_worker_mapping[Role.VerifierCritic] = ray.remote(VerifierCriticWorker)
                role_worker_mapping[Role.VerifierRefPolicy] = ray.remote(VerifierActorRolloutWorker)
                
        # set up resource pools
        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        
        # map roles to resource pools
        role_to_pool_mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        # add verifier roles to resource pool mapping
        if config.verifier.enable:
            role_to_pool_mapping[Role.VerifierActorRollout] = global_pool_id
            
            if config.verifier.get('trainable', False):
                role_to_pool_mapping[Role.VerifierCritic] = global_pool_id
                role_to_pool_mapping[Role.VerifierRefPolicy] = global_pool_id
            
            
        if config.reward_model.get('enable', False):
            if config.reward_model.strategy == 'fsdp':
                from verl.workers.reward_worker import CustomizedRewardModelWorker as RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            role_to_pool_mapping[Role.RewardModel] = global_pool_id

        # create resource pool manager
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, 
            mapping=role_to_pool_mapping
        )
        
        # set up generator reward manager/function
        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name == 'naive':
            from verl.workers.reward_manager import NaiveRewardManager
            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == 'prime':
            from verl.workers.reward_manager import PrimeRewardManager
            reward_manager_cls = PrimeRewardManager
        else:
            raise NotImplementedError
        
        gen_compute_score = get_custom_reward_fn(config)
        
        correct_reward_fn = reward_manager_cls(tokenizer=gen_tokenizer, num_examine=0, compute_score=gen_compute_score)
        customize_format_fn = reward_manager_cls(tokenizer=gen_tokenizer, num_examine=1, compute_score=format_compute_score)

        verifier_reward_fn = None

        trainer = RLTangoTrainer(
            config=config,
            tokenizer=gen_tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn_list=[correct_reward_fn, customize_format_fn], # this is the reward function list for generator
            verifier_tokenizer=verifier_tokenizer,
            verifier_reward_fn=verifier_reward_fn # this is the reward function for verifier, if enabled
        )
        
        trainer.init_workers()        
        trainer.fit()

if __name__ == '__main__':
    main() 