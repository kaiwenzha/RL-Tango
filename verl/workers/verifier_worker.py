import re
import torch
from verl import DataProto
from verl.single_controller.base.decorator import register, Dispatch, Execute
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
from verl.utils import torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.reward_score.format import verifier_format_reward
from data_preprocess.system_prompt import VERIFIER_PROMPT_TEMPLATE


class VerifierActorRolloutWorker(ActorRolloutRefWorker):
    """
    Worker for the verifier model that combines actor and rollout functionality.
    This is a specialized version of ActorRolloutRefWorker for verification tasks.
    """

    def __init__(self, config, role='actor_rollout'):
        super().__init__(config, role)
        
    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def prepare_verification_input(self, data: DataProto, generator_tokenizer, n_rollouts=1):
        """
        Prepare input for verification from generator data by adding verification instructions and step positions.
        
        Args:
            data: DataProto with generator input and outputs
            generator_tokenizer: Tokenizer for generator model
            n_rollouts: Number of rollouts to generate
            
        Returns:
            DataProto with formatted verification input
        """
        
        # get original questions and generator responses
        generator_input_ids = data.batch['input_ids']
        generator_responses = data.batch['responses']
        batch_size = generator_input_ids.shape[0]
        device = generator_input_ids.device
        
        # try to extract clean questions from extra_info if available
        generator_clean_questions = []
        if 'raw_question' in data.non_tensor_batch:
            assert len(data.non_tensor_batch['raw_question']) == batch_size, "Mismatch in batch size and raw_question length"
            generator_clean_questions.extend(data.non_tensor_batch['raw_question'])
        else:
            raise RuntimeError("raw_question is not in data.non_tensor_batch")

        # extract and store step positions in generator responses for reward mapping
        step_positions = []
        generator_clean_responses = []
        
        for i in range(batch_size):
            r_tokens = generator_responses[i]
            mask = r_tokens != generator_tokenizer.pad_token_id
            valid_tokens = r_tokens[mask]
            
            if len(valid_tokens) > 0:
                generator_response_text = generator_tokenizer.decode(valid_tokens, skip_special_tokens=True)

                steps_positions_current = []
                for match in re.finditer(r'<step>(.*?)</step>', generator_response_text, re.DOTALL):
                    steps_positions_current.append(match.span())

                generator_clean_responses.append(generator_response_text)
                step_positions.append(steps_positions_current)
            else:
                generator_clean_responses.append("")
                step_positions.append([])

        # store step positions and original generator responses
        # it is werid that we must store it seperately and return it, instead of saving it to the `output_batch`
        temp_storage = {
            'step_positions': step_positions,
            'step_counts': [len(steps) for steps in step_positions],
        }
        data.meta_info.update(temp_storage)
        
        # prepare verification inputs
        verification_inputs = []
        for generator_question, generator_response, step_position in zip(generator_clean_questions, generator_clean_responses, step_positions):
            
            verification_input = VERIFIER_PROMPT_TEMPLATE.format(
                problem=generator_question,
                solution=generator_response,
                generator_step_count=len(step_position)
            )
            verification_inputs.append(verification_input)
        
        # after preparing verification inputs, duplicate them for bootstrapping if requested
        if n_rollouts > 1:
            bootstrapped_verification_inputs = []
            for inp in verification_inputs:
                bootstrapped_verification_inputs.extend([inp] * n_rollouts)

            verification_inputs = bootstrapped_verification_inputs
        
        # process each verification input separately and then aggregate
        all_input_ids = []
        all_attention_masks = []

        for ver_input in verification_inputs:
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=[ver_input],
                tokenizer=self.tokenizer,
                max_length=self.config.rollout.prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation='right'
            )
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)

        # concatenate results
        input_ids = torch.cat(all_input_ids, dim=0)
        attention_mask = torch.cat(all_attention_masks, dim=0)

        # generate position IDs after concatenation
        position_ids = compute_position_id_with_mask(attention_mask)

        verifier_tokenizer_outputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
        
        # create output batch and store metadata
        output_batch = DataProto.from_single_dict(verifier_tokenizer_outputs).to(device)
        output_batch.meta_info = {
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'temperature': 1.0,
            'top_p': 1.0,
            'do_sample': True,
        }
        
        # override defaults if rollout exists in the config
        if hasattr(self.config, 'rollout'):
            rollout_config = self.config.rollout
            for param in ['temperature', 'top_p', 'do_sample']:
                if hasattr(rollout_config, param):
                    output_batch.meta_info[param] = getattr(rollout_config, param)
        
        return output_batch, temp_storage
    
    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def extract_verification_result(self, data: DataProto, original_data: DataProto, n_rollouts=1):
        """
        Extract structured verification results from model outputs.
        
        Args:
            data: DataProto with verification generation
            original_data: Original DataProto with generator outputs
            n_rollouts: Number of rollouts to generate
            
        Returns:
            DataProto with structured verification results
        """
        generator_responses = original_data.batch.get('responses')
        verifier_responses = data.batch.get('responses')
        
        if generator_responses is None or verifier_responses is None:
            raise ValueError("Missing responses in input data")
                
        verifier_batch_size = verifier_responses.shape[0]
        generator_batch_size = generator_responses.shape[0]
        
        bootstrapping = n_rollouts > 1
        
        # if bootstrapping, each generator example was replicated n_rollouts times in the verifier batch
        if bootstrapping:
            if verifier_batch_size != generator_batch_size * n_rollouts:
                generator_batch_size = verifier_batch_size // n_rollouts
        
        generator_response_length = generator_responses.shape[1]
        verifier_response_length = verifier_responses.shape[1]
        device = verifier_responses.device
        
        # retrieve step positions and counts from original data
        assert 'temp_storage' in original_data.meta_info, "temp_storage is missing in original_data.meta_info"
        step_positions = original_data.meta_info['temp_storage'].get('step_positions', [[]] * generator_batch_size)
        step_counts = original_data.meta_info['temp_storage'].get('step_counts', [0] * generator_batch_size)
        G_format_reward = original_data.meta_info['temp_storage'].get('G_format_reward', [0] * generator_batch_size)
        
        # create results tensors for generator output size
        ## All the following variables are initialized as all-zero tensors, and this has meaningful implications. 
        ## For example, V_correctness_reward is initialized as if all verifier judgments are incorrect by default. 
        ## This is important because in our code, if `final_match` is False, we skip updating V_correctness_reward. 
        ## Similarly, `verification_label` is also initialized to False by default, meaning the verifier is assumed to mark all generator responses as incorrect initially.
        ## In rare corner cases (e.g., when `final_match` is False but the generator response is actually incorrect), this can lead to slight inconsistencies.
        ## That said, this rarely impacts training if the verifier generally follows instructions well.

        G_step_rewards = torch.zeros((generator_batch_size, n_rollouts, generator_response_length), dtype=torch.float, device=device)
        G_step_rewards_mask = torch.zeros((generator_batch_size, n_rollouts, generator_response_length), dtype=torch.bool, device=device)
        G_correctness_reward_from_verifier = torch.zeros((generator_batch_size, n_rollouts, generator_response_length), dtype=torch.float, device=device)
        G_final_match_or_not = torch.zeros((generator_batch_size, n_rollouts), dtype=torch.float, device=device)
        V_correctness_reward = torch.zeros((generator_batch_size, verifier_response_length), dtype=torch.float, device=device)
        V_format_reward = torch.zeros((generator_batch_size, verifier_response_length), dtype=torch.float, device=device)
        V_reward_mask = torch.zeros((generator_batch_size, verifier_response_length), dtype=torch.bool, device=device)
        verification_label = torch.zeros((generator_batch_size,), dtype=torch.float, device=device)
        
        # get valid response lengths
        valid_generator_lengths = original_data.batch['attention_mask'][:, -generator_responses.size(1):].sum(dim=1)
        valid_verifier_lengths = data.batch['attention_mask'][:, -verifier_responses.size(1):].sum(dim=1)
        
        # compile regex patterns once outside the loop
        step_verification_pattern = re.compile(r'<step_verification>(.*?)</step_verification>', re.DOTALL)
        final_verification_pattern = re.compile(r'<final_verification>\\box(?:ed)?\{(CORRECT|INCORRECT)\}(?:\.?\s*)?</final_verification>', re.IGNORECASE)
        step_pattern = re.compile(r'<step>(.*?)\\box(?:ed)?\{(CORRECT|INCORRECT)\}(?:\.?\s*)?</step>', re.IGNORECASE | re.DOTALL)
        
        # store first rollout responses and all verification results
        first_rollout_verifier_response_texts = []
        all_verification_results = []
        
        # process all responses
        for i in range(verifier_batch_size):
            # calculate the generator example index and rollout index
            if bootstrapping:
                example_idx = i // n_rollouts
                rollout_idx = i % n_rollouts
            else:
                example_idx = i
                rollout_idx = 0
            
            # decode verifier response
            if valid_verifier_lengths[i] > 0:
                valid_tokens = verifier_responses[i, :valid_verifier_lengths[i]]
                verifier_response_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
            else:
                verifier_response_text = ""
            
            # store first rollout's responses for output
            if rollout_idx == 0:
                first_rollout_verifier_response_texts.append(verifier_response_text)
            
            # extract step verifications
            step_verification_match = step_verification_pattern.search(verifier_response_text)
            step_verification_content = step_verification_match.group(1) if step_verification_match else ""
            
            # extract step judgments
            step_matches = step_pattern.findall(step_verification_content)
            step_judgments = [judgment.upper() == "CORRECT" for content, judgment in step_matches]
            
            # extract final verification
            final_match = final_verification_pattern.search(verifier_response_text)
            final_result = False
            if final_match:
                final_result = final_match.group(1).upper() == "CORRECT"
                
                G_final_match_or_not[example_idx, rollout_idx] = 1.0
                G_correctness_reward_from_verifier[example_idx, rollout_idx, valid_generator_lengths[example_idx]-1] = 1.0 if final_result else 0.0
                
                # for first rollout (or non-bootstrapping), compute verifier rewards
                if rollout_idx == 0:
                    verification_label[example_idx] = 1.0 if final_result else 0.0
                    
                    # verification correctness reward
                    ground_truth_correctness = data.batch['ground_truth_correctness'][i]
                    is_correct_verification = (verification_label[example_idx] > 0.5) == (ground_truth_correctness > 0.5)
                    V_correctness_reward[example_idx, valid_verifier_lengths[i]-1] = is_correct_verification.float()
                    
                    # format reward
                    V_format_reward[example_idx, valid_verifier_lengths[i]-1] = verifier_format_reward(
                        verifier_response_text, 
                        generator_steps_count=step_counts[example_idx], 
                        generator_format_score=G_format_reward[example_idx]
                    )
                    V_reward_mask[example_idx, valid_verifier_lengths[i]-1] = 1
            # process step judgments and assign rewards
            expected_step_count = step_counts[example_idx]
            generator_step_position = step_positions[example_idx]
            
            if valid_generator_lengths[example_idx] > 0 and len(step_judgments) == expected_step_count and expected_step_count > 0:
                # map step judgments to token positions
                generator_response = generator_responses[example_idx]
                generator_response_text = self.tokenizer.decode(
                    generator_response[:valid_generator_lengths[example_idx]], 
                    skip_special_tokens=True
                )
                
                # create character-to-token mapping
                encoding = self.tokenizer(
                    generator_response_text, 
                    return_offsets_mapping=True, 
                    add_special_tokens=False
                )
                offset_mapping = encoding.offset_mapping
                
                # process each step
                for step_idx, (step_start, step_end) in enumerate(generator_step_position):
                    step_judgment = step_judgments[step_idx] if step_idx < len(step_judgments) else False
                    
                    # map character positions to token indices
                    token_start, token_end = None, None
                    
                    # find token indices that contain these character positions
                    for token_idx, (char_start, char_end) in enumerate(offset_mapping):
                        if char_start <= step_start < char_end and token_start is None:
                            token_start = token_idx
                        if char_start <= step_end < char_end:
                            token_end = token_idx
                            break
                    
                    # default to beginning/end if not found
                    if token_start is None:
                        token_start = 0
                    if token_end is None:
                        token_end = len(offset_mapping) - 1
                    
                    # ensure token indices are within bounds
                    token_start = min(token_start, valid_generator_lengths[example_idx] - 1)
                    token_end = min(token_end, valid_generator_lengths[example_idx] - 1)
                    
                    # apply reward based on the configured strategy
                    if token_start <= token_end:
                        step_reward = (1.0 if step_judgment else -1.0) / expected_step_count
                        G_step_rewards[example_idx, rollout_idx, token_end] = step_reward
                        G_step_rewards_mask[example_idx, rollout_idx, token_end] = 1
            
            # store verification results (for first rollout only)
            if rollout_idx == 0:
                result = {
                    "numbered_judgments": step_judgments,
                    "generator_step_count": expected_step_count,
                    "verifier_step_count": len(step_judgments)
                }
                all_verification_results.append(result)
        
        # calculate alignment metrics (using first rollout only)
        per_example_alignment = []
        for res in all_verification_results:
            gen_count = res['generator_step_count']
            ver_count = res['verifier_step_count']
            per_example_alignment.append({
                'is_perfect': gen_count == ver_count,
                'has_missing': gen_count > ver_count,
                'has_extra': ver_count > gen_count
            })

        # calculate overall rates for this batch
        alignment_rates = {
            "perfect_alignment_rate": sum(ex['is_perfect'] for ex in per_example_alignment) / generator_batch_size if generator_batch_size > 0 else 0,
            "missing_steps_rate": sum(ex['has_missing'] for ex in per_example_alignment) / generator_batch_size if generator_batch_size > 0 else 0,
            "extra_steps_rate": sum(ex['has_extra'] for ex in per_example_alignment) / generator_batch_size if generator_batch_size > 0 else 0
        }
        
        # create output batch with results
        output_batch = DataProto.from_dict(
        tensors = {
            'verification_label': verification_label,   # (B)
            'G_step_rewards': G_step_rewards,   # (B, n_rollouts, L)
            'G_step_rewards_mask': G_step_rewards_mask,   # (B, n_rollouts, L)
            'G_correctness_reward_from_verifier': G_correctness_reward_from_verifier,   # (B, n_rollouts, L)
            'G_final_match_or_not': G_final_match_or_not,   # (B, n_rollouts)
            'V_correctness_reward': V_correctness_reward,   # (B, L')
            'V_format_reward': V_format_reward,   # (B, L')
            'V_reward_mask': V_reward_mask,   # (B, L')
        },
        non_tensors = {
            'per_example_alignment': per_example_alignment,
            'verifier_responses': first_rollout_verifier_response_texts,
        },
        meta_info={
            'alignment_rates': alignment_rates,
        })

        return output_batch
    
class VerifierCriticWorker(CriticWorker):
    """Worker for the verifier's critic model."""
    
    def __init__(self, config):
        super().__init__(config)
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        return super().compute_values(data)
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        return super().update_critic(data) 