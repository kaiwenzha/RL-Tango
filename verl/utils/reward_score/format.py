import re

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None
    
    
def calculate_step_tag_penalty(text):
    penalty = 0
    tag_pattern = re.compile(r'</?step>')


    tags = list(tag_pattern.finditer(text))
    stack = []

    for match in tags:
        tag = match.group()
        pos = match.start()

        if tag == '<step>':
            if stack:
                penalty += 1
            else:
                stack.append(pos)
        elif tag == '</step>':
            if not stack:
                penalty += 1
            else:
                stack.pop()

    return penalty

def generator_format_reward(solution_str):
    solution_str = solution_str.rstrip()
    answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    if not answer_match:
        return 0.0
    
    score = 1.0
    penalties = []
    
    think_content = solution_str
    answer_content = answer_match.group(1).strip()
    
    if not remove_boxed(answer_content):
        penalties.append(("Answer not in \\boxed{} format", 0.8))
    
    step_matches = re.findall(r'<step>(.*?)</step>', think_content, re.DOTALL)
    
    if not step_matches:
        penalties.append(("No <step></step> tags found", 0.8))
    else:
        if len(step_matches) < 3:
            penalties.append((f"Too few steps ({len(step_matches)})", 0.4))

    for _, penalty in penalties:
        score -= penalty

    return max(0.0, min(1.0, score))

def format_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == 'openai/gsm8k':
        res = 0.0
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        raise NotImplementedError
    elif data_source in ['MATH500', 'AIME2024', 'AIME2025', 'AMC2023', 'OlympiadBench', 'BGQA', 'CRUXEval', 'StrategyQA', 'TableBench', 'Eurus-2-RL-Data']:
        res = generator_format_reward(solution_str)
    else:
        raise NotImplementedError

    return res

def verifier_format_reward(verifier_response_text, generator_steps_count=None, generator_format_score=1.0):
    verifier_response_text = verifier_response_text.strip()
    
    base_pattern = r"^<step_verification>.*?</step_verification>\s*<final_verification>.*?</final_verification>$"
    base_match = re.match(base_pattern, verifier_response_text, re.DOTALL)
    
    if not base_match:
        return 0.0
    
    step_verification_match = re.search(r'<step_verification>(.*?)</step_verification>', verifier_response_text, re.DOTALL)
    final_verification_match = re.search(r'<final_verification>(.*?)</final_verification>', verifier_response_text, re.DOTALL)
    
    if not step_verification_match or not final_verification_match:
        return 0.0
    
    score = 1.0
    penalties = []
    
    step_verification_content = step_verification_match.group(1).strip()
    final_verification_content = final_verification_match.group(1).strip()
    
    final_verification_pattern = r'\\box(?:ed)?\{(CORRECT|INCORRECT)\}(?:\.?\s*)?$'
    final_match = re.search(final_verification_pattern, final_verification_content, re.IGNORECASE)
    if not final_match:
        penalties.append((f"Final verification not in \\box{{CORRECT}} or \\box{{INCORRECT}} format, content = {final_verification_content}", 0.8))
    
    step_matches = re.findall(r'<step>(.*?)</step>', step_verification_content, re.DOTALL)
    
    if not step_matches:
        penalties.append(("No <step></step> tags found in verification", 0.8))
    else:
        valid_step_count = 0
        for i, step_content in enumerate(step_matches):
            step_judgment_pattern = r'\\box(?:ed)?\{(CORRECT|INCORRECT)\}(?:\.?\s*)?$' 
            step_judgment = re.search(step_judgment_pattern, step_content, re.IGNORECASE)
            if step_judgment:
                valid_step_count += 1
        
        if generator_steps_count is not None and generator_format_score >= 0.6:
            if valid_step_count != generator_steps_count:
                penalties.append((f"Mismatch in step count: {valid_step_count} verifications for {generator_steps_count} generator steps", 0.8))
                
    content_without_steps = re.sub(r'<step>.*?</step>', '', step_verification_content, flags=re.DOTALL)
    if re.sub(r'\s+', '', content_without_steps):
        penalties.append(("Content outside <step> tags inside <step_verification>", 0.4))
    
    for _, penalty in penalties:
        score -= penalty

    return max(0.0, min(1.0, score))