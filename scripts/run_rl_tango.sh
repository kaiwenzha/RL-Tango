#!/bin/bash
set -x

if [ -z "$1" ]; then
    echo "Error: GENERATOR_MODEL_PATH is required"
    echo "Usage: $0 GENERATOR_MODEL_PATH"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=./logs/run_${TIMESTAMP}
mkdir -p $LOG_DIR

export VERL_DEBUG_LEVEL=DEBUG
export VERL_LOG_FILE=$LOG_DIR/rltango_training.log
export PYTHONUNBUFFERED=1

train_files="['./data/eurus2_rl_math/train.parquet']"

test_root_path=./data/eval_benchmarks
test_data_sources=(
    "MATH500"
    "AIME2024"
    "AIME2025"
    "AMC2023"
    "OlympiadBench"
    "BGQA"
    "CRUXEval"
    "StrategyQA"
    "TableBench"
)

test_files="["
if [ ${#test_data_sources[@]} -gt 0 ]; then
    first_source=${test_data_sources[0]}
    first_path=$test_root_path/$first_source/test.parquet
    test_files="$test_files'$first_path'"
fi
for data_source in "${test_data_sources[@]:1}"; do
    test_path=$test_root_path/$data_source/test.parquet
    test_files="$test_files, '$test_path'"
done
test_files="$test_files]"

GENERATOR_MODEL_PATH=$1
VERIFIER_MODEL_PATH=./base_models/Qwen2.5-7B

VERIFIER_ENABLE=True
VERIFIER_TRAINABLE=True

# RL Algorithms
ADV_ESTIMATOR=grpo
if [ "$ADV_ESTIMATOR" = "grpo" -o "$ADV_ESTIMATOR" = "rloo" ]; then
    # GRPO/RLOO settings
    ROLLOUT_N=5
    USE_KL_LOSS=True
    KL_LOSS_COEF=0.001
    KL_LOSS_TYPE=low_var_kl
    PPO_MINI_BATCH_SIZE=256
    PPO_MICRO_BATCH_SIZE_PER_GPU=4
    LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=16
    TENSOR_MODEL_PARALLEL_SIZE=2
    KL_CTRL_KL_COEF=0.001
    ENTROPY_COEFF=0.001
    GPU_MEMORY_UTILIZATION=0.6
    ALPHA_START=0.1
elif [ "$ADV_ESTIMATOR" = "reinforce_plus_plus" ]; then
    # Reinforce++ setting
    ROLLOUT_N=5
    USE_KL_LOSS=True
    KL_LOSS_COEF=0.001
    KL_LOSS_TYPE=mse
    PPO_MINI_BATCH_SIZE=256
    PPO_MICRO_BATCH_SIZE_PER_GPU=4
    LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=16
    TENSOR_MODEL_PARALLEL_SIZE=2
    KL_CTRL_KL_COEF=0.001
    ENTROPY_COEFF=0.0
    GPU_MEMORY_UTILIZATION=0.6
    ALPHA_START=0.5
fi

NNODES=4
N_GPUS_PER_NODE=8
TRAIN_BATCH_SIZE=256
ENABLE_GRAD_CHECKPOINTING=True
USE_REMOVE_PADDING=True
MAX_NUM_BATCHED_TOKENS=8192
APPLY_CHAT_TEMPLATE=False
VAL_BEFORE_TRAIN=True

# Learning rates
GENERATOR_LR=1e-6
CRITIC_LR=1e-5
VERIFIER_LR=1e-6
VERIFIER_CRITIC_LR=1e-5
ACTOR_LR_WARMUP_STEPS_RATIO=0.0
CRITIC_LR_WARMUP_STEPS_RATIO=0.0
VERIFIER_ACTOR_LR_WARMUP_STEPS_RATIO=0.0
VERIFIER_CRITIC_LR_WARMUP_STEPS_RATIO=0.0

# prompt/response length
GENERATOR_MAX_PROMPT_LENGTH=2048
GENERATOR_MAX_RESPONSE_LENGTH=2048
VERIFIER_MAX_PROMPT_LENGTH=4096
VERIFIER_MAX_RESPONSE_LENGTH=4096

TOTAL_EPOCHS=15  # deprecated
TOTAL_TRAINING_STEPS=440
SAVE_FREQ=50
TEST_FREQ=10
CRITIC_WARMUP=0
VERIFIER_CRITIC_WARMUP=0
N_GENERATOR_STEPS=3
N_VERIFIER_STEPS=1
GENERATOR_WARMUP_STEPS=0
VERIFIER_WARMUP_STEPS=40
N_VERIFIER_INFERENCE_ROLLOUTS=1
VERIFIER_TEMPERATURE=1.0

GROUND_TRUTH_WEIGHT=1.0
GENERATOR_FORMAT_REWARD_WEIGHT=0.0
VERIFIER_FORMAT_REWARD_WEIGHT=0.8
VERIFIER_CORRECTNESS_REWARD_WEIGHT=1.0
GENERATOR_OUTCOME_REWARD_WEIGHT=1.0
GENERATOR_PROCESS_REWARD_WEIGHT=1.0
ALPHA_SCHEDULE='exp'
ALPHA_MIN=1e-3
EXP_TARGET_FRAC=1.0

VERIFIER_LABEL_EMA_DECAY=0.8
REWEIGHT_VERIFIER_REWARDS=True
REWEIGHT_METHOD='sqrt_inverse'

PROJECT_NAME='RL-Tango'
EXPERIMENT_NAME='rl-tango-training'

# Environment settings
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_rl_tango \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    verifier_algorithm.adv_estimator=$ADV_ESTIMATOR \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$GENERATOR_MAX_PROMPT_LENGTH \
    data.max_response_length=$GENERATOR_MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.apply_chat_template=$APPLY_CHAT_TEMPLATE \
    actor_rollout_ref.model.path=$GENERATOR_MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRAD_CHECKPOINTING \
    actor_rollout_ref.model.use_remove_padding=$USE_REMOVE_PADDING \
    actor_rollout_ref.actor.optim.lr=$GENERATOR_LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$ACTOR_LR_WARMUP_STEPS_RATIO \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    critic.optim.lr=$CRITIC_LR \
    critic.optim.lr_warmup_steps_ratio=$CRITIC_LR_WARMUP_STEPS_RATIO \
    critic.model.use_remove_padding=$USE_REMOVE_PADDING \
    critic.model.path=$GENERATOR_MODEL_PATH \
    critic.model.enable_gradient_checkpointing=$ENABLE_GRAD_CHECKPOINTING \
    critic.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=$KL_CTRL_KL_COEF \
    algorithm.ground_truth_weight=$GROUND_TRUTH_WEIGHT \
    algorithm.generator_format_reward_weight=$GENERATOR_FORMAT_REWARD_WEIGHT \
    algorithm.generator_outcome_reward_weight=$GENERATOR_OUTCOME_REWARD_WEIGHT \
    algorithm.generator_process_reward_weight=$GENERATOR_PROCESS_REWARD_WEIGHT \
    algorithm.alpha_schedule=$ALPHA_SCHEDULE \
    algorithm.alpha_start=$ALPHA_START \
    algorithm.alpha_min=$ALPHA_MIN \
    algorithm.exp_target_frac=$EXP_TARGET_FRAC \
    verifier.enable=$VERIFIER_ENABLE \
    verifier.trainable=$VERIFIER_TRAINABLE \
    verifier.model.path=$VERIFIER_MODEL_PATH \
    verifier.model.enable_gradient_checkpointing=$ENABLE_GRAD_CHECKPOINTING \
    verifier.model.use_remove_padding=$USE_REMOVE_PADDING \
    verifier.actor.optim.lr=$VERIFIER_LR \
    verifier.actor.optim.lr_warmup_steps_ratio=$VERIFIER_ACTOR_LR_WARMUP_STEPS_RATIO \
    verifier.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    verifier.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    verifier.actor.use_kl_loss=$USE_KL_LOSS \
    verifier.actor.kl_loss_coef=$KL_LOSS_COEF \
    verifier.actor.kl_loss_type=$KL_LOSS_TYPE \
    verifier.actor.entropy_coeff=$ENTROPY_COEFF \
    verifier.actor.fsdp_config.param_offload=False \
    verifier.actor.fsdp_config.optimizer_offload=False \
    verifier.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    verifier.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
    verifier.rollout.name=vllm \
    verifier.rollout.prompt_length=$VERIFIER_MAX_PROMPT_LENGTH \
    verifier.rollout.response_length=$VERIFIER_MAX_RESPONSE_LENGTH \
    verifier.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    verifier.rollout.max_num_batched_tokens=12800 \
    verifier.rollout.n_verifier_inference_rollouts=$N_VERIFIER_INFERENCE_ROLLOUTS \
    verifier.rollout.temperature=$VERIFIER_TEMPERATURE \
    verifier.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    verifier.ref.fsdp_config.param_offload=True \
    verifier.rollout.n=$ROLLOUT_N \
    verifier_critic.model.path=$VERIFIER_MODEL_PATH \
    verifier_critic.model.enable_gradient_checkpointing=$ENABLE_GRAD_CHECKPOINTING \
    verifier_critic.model.use_remove_padding=$USE_REMOVE_PADDING \
    verifier_critic.optim.lr=$VERIFIER_CRITIC_LR \
    verifier_critic.optim.lr_warmup_steps_ratio=$VERIFIER_CRITIC_LR_WARMUP_STEPS_RATIO \
    verifier_critic.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    verifier_critic.model.fsdp_config.param_offload=False \
    verifier_critic.model.fsdp_config.optimizer_offload=False \
    verifier_algorithm.kl_ctrl.kl_coef=$KL_CTRL_KL_COEF \
    verifier_algorithm.verifier_correctness_reward_weight=$VERIFIER_CORRECTNESS_REWARD_WEIGHT \
    verifier_algorithm.verifier_format_reward_weight=$VERIFIER_FORMAT_REWARD_WEIGHT \
    verifier_algorithm.verifier_label_ema_decay=$VERIFIER_LABEL_EMA_DECAY \
    verifier_algorithm.reweight_verifier_rewards=$REWEIGHT_VERIFIER_REWARDS \
    verifier_algorithm.reweight_method=$REWEIGHT_METHOD \
    trainer.generator_warmup_steps=$GENERATOR_WARMUP_STEPS \
    trainer.verifier_warmup_steps=$VERIFIER_WARMUP_STEPS \
    trainer.critic_warmup=$CRITIC_WARMUP \
    trainer.verifier_critic_warmup=$VERIFIER_CRITIC_WARMUP \
    trainer.n_generator_steps=$N_GENERATOR_STEPS \
    trainer.n_verifier_steps=$N_VERIFIER_STEPS \
    trainer.initial_mode=generator \
    trainer.logger=['wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.val_generations_to_log_to_wandb=30 \
    trainer.val_verifications_to_log_to_wandb=30 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.resume_mode=auto \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    2>&1 | tee -a $LOG_DIR/full_output.log