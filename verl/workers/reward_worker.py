from verl.workers.fsdp_workers import RewardModelWorker, copy_to_local, hf_tokenizer, get_sharding_strategy
import warnings
import torch 
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)

# Customized RewardModelWorker to support AutoModelForSequenceClassification instead of original RewardModelWorker's AutoModelForTokenClassification
class CustomizedRewardModelWorker(RewardModelWorker):
    def _build_model(self, config):
            # the following line is necessary
            from torch.distributed.fsdp import CPUOffload
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from transformers import AutoConfig, AutoModelForSequenceClassification

            # download the checkpoint from hdfs
            local_path = copy_to_local(config.model.path)

            self._do_switch_chat_template = False
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

            trust_remote_code = config.model.get("trust_remote_code", False)
            model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

            self.gain = config.gain # config.get("gain", 1.0)
            self.bias = config.bias # config.get("bias", 0.0)

            # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
            init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh)

            with init_context(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_config.classifier_dropout = 0.0
                reward_module = AutoModelForSequenceClassification.from_pretrained(
                    pretrained_model_name_or_path=local_path,
                    config=model_config,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=trust_remote_code,
                )

                if config.model.get("use_remove_padding", False) or self.ulysses_sequence_parallel_size > 1:
                    from verl.models.transformers.monkey_patch import apply_monkey_patch

                    apply_monkey_patch(model=reward_module, ulysses_sp_size=self.ulysses_sequence_parallel_size)

                reward_module.to(torch.bfloat16)

            auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

            fsdp_mesh = self.device_mesh
            sharding_strategy = get_sharding_strategy(fsdp_mesh)

            if config.strategy == "fsdp":
                reward_module = FSDP(
                    reward_module,
                    param_init_fn=init_fn,
                    use_orig_params=False,
                    auto_wrap_policy=auto_wrap_policy,
                    device_id=torch.cuda.current_device(),
                    sharding_strategy=sharding_strategy,  # zero3
                    sync_module_states=True,
                    cpu_offload=CPUOffload(offload_params=True),
                    forward_prefetch=False,
                    device_mesh=self.device_mesh,
                )
            else:
                raise NotImplementedError(f"Unknown strategy: {config.strategy}")
            return reward_module 

    def _forward_micro_batch(self, micro_batch):
        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input

        from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            if self.use_remove_padding:
                # this if branch is not carefully checked/tested, might have bugs due to tensor shapes when calculating reward_rmpad, rm_score. 
                # Compare with the original verl.workers.fsdp_workers.RewardModelWorker for more details
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False)  # prevent model thinks we are generating
                reward_rmpad = output.logits.view(1,1,1)
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outpus_and_unpad(reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
                rm_score = rm_score.view(-1)
            else:
                output = self.reward_module(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
                rm_score = output.logits.view(-1)

            rm_score = rm_score * self.gain + self.bias
            return rm_score