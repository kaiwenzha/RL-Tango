import re
import math
from verl.utils.reward_score.prime_math import match_answer, grade_answer, math_equal

def compute_score(solution_str, ground_truth):
    # solution_str = extract_solution(solution_str)
    _, answer = match_answer(solution_str)
    if grade_answer(answer, ground_truth):
        return 1.0
    try:
        if "\pi" in answer or "\pi" in ground_truth:
            equivs = []
            for pi in [math.pi, 3.14]:
                equivs.append(math_equal(answer, ground_truth, timeout=True, pi=pi))
            is_correct = any(equivs)
        else:
            is_correct = math_equal(answer, ground_truth, timeout=True)
    except:
        is_correct = False
    if is_correct:
        return 1.0
    else:
        return 0.0
        
def extract_solution(solution_str):
    # pattern = re.compile(r"(\\boxed{.*})")
    pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
    matches = re.findall(pattern, solution_str)
    return matches[-1] if matches else ""