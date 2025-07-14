import re
from verl.utils.reward_score.math_utils import solution2answer, is_equal

def compute_score(solution_str, ground_truth):
    if is_equal((solution2answer(ground_truth)), (solution2answer(solution_str))):
        return 1.0
    else:
        return 0.0
        
def extract_solution(solution_str):
    pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
    matches = re.findall(pattern, solution_str)
    return matches[-1] if matches else ""