import ast
from verl.utils.reward_score.math_utils import solution2answer

def is_equiv(str1, str2, verbose=False):
    def load_obj_as_str(s):
        try:
            return ast.literal_eval(s)
        except:
            return s

    answer_a = load_obj_as_str(str1)
    answer_b = load_obj_as_str(str2)
    return answer_a == answer_b


def compute_score(solution_str, ground_truth) -> float:
    final_answer = solution2answer(solution_str)
    if final_answer is None or final_answer.strip() == "":
        return 0.0
    
    if is_equiv(final_answer, ground_truth):
        return 1.0
    else:
        return 0.0    