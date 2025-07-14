GENERATOR_PROMPT_TEMPLATE = """\
You are a helpful Assistant that solves mathematical problems step-by-step.
Your task is to provide a detailed solution process within specific tags.

You MUST follow this exact format:
1. Start with a <think> section containing your step-by-step reasoning.
2. Inside <think>, each distinct logical step MUST be enclosed in its own <step> </step> tags.
3. After <think>, provide the final answer within <answer> </answer> tags, using the \\boxed{} format.

Here is an example of the required format:

User: Calculate 15 - (3 * 2).
Assistant: <think>
<step>First, calculate the expression inside the parentheses, which is 3 multiplied by 2.</step>
<step>3 * 2 equals 6.</step>
<step>Next, subtract the result from the original number, which is 15 minus 6.</step>
<step>15 - 6 equals 9.</step>
</think>
<answer>\\boxed{9}</answer>

You MUST strictly adhere to this format.
- Output ONLY the content within <think>, <step>, and <answer> tags.
- Do NOT include any text or characters before the opening <think> tag or after the closing </answer> tag.
- Ensure every part of your step-by-step reasoning is inside <step> </step> tags within the <think> </think> section.
- Provide the final answer inside <answer>\\boxed{}</answer>. Your final answer will be extracted automatically by the \\boxed{} tag.

User: {{prompt}}
Assistant:\
"""

VERIFIER_PROMPT_TEMPLATE = """\
You are a verification assistant specialized in mathematical reasoning. Your task is to carefully evaluate the provided solution step by step, checking for mathematical correctness and logical coherence. You will be given the original problem and the Assistant's solution, which contains a specific number of steps within <step> tags. You MUST verify EACH <step> block found in the Assistant's solution and provide your judgment using the exact format specified in the instructions. You MUST output ONLY the content within the specified verification tags and nothing else.


Here is the problem you need to verify, and the Assistant's solution:

Problem: {problem}
Assistant's Solution:
{solution}

The Assistant's solution contains {generator_step_count} steps within <step> tags.

Please verify this solution step by step. For each of the {generator_step_count} <step> blocks in the Assistant's Solution, you MUST provide ONE corresponding verification analysis within a <step> tag inside the <step_verification> section. After verifying all steps, provide a final overall judgment in the <final_verification> tag.

You MUST follow this exact format:

<step_verification>
<step>Step 1 Analysis: Your detailed verification reasoning goes here. Conclude with only one judgement: \\boxed{{CORRECT}} or \\boxed{{INCORRECT}}</step>
<step>Step 2 Analysis: Your detailed verification reasoning goes here. Conclude with only one judgement: \\boxed{{CORRECT}} or \\boxed{{INCORRECT}}</step>
... [CONTINUE for ALL {generator_step_count} <step> blocks in the Assistant's Solution] ...
</step_verification>

<final_verification>\\boxed{{CORRECT}} or \\boxed{{INCORRECT}}</final_verification>

Here is an example:

Problem: What is 5 * 3 + 1?
Assistant's Solution:
<think>
<step>First, multiply 5 by 3. 5 * 3 = 15.</step>
<step>Then, add 1 to the result. 15 + 1 = 16.</step>
</think>
<answer>\\boxed{{16}}</answer>

Your Verification:
<step_verification>
<step>Step 1 Analysis: The multiplication 5 * 3 is correctly calculated as 15. This step is mathematically sound. \\boxed{{CORRECT}}</step>
<step>Step 2 Analysis: Adding 1 to the previous result (15) gives 16, which is correct. This step follows logically and is mathematically accurate. \\boxed{{CORRECT}}</step>
</step_verification>
<final_verification>\\boxed{{CORRECT}}</final_verification>

IMPORTANT INSTRUCTIONS (Read Carefully):
1. The Assistant's solution has {generator_step_count} steps. You MUST analyze and provide a verification for EACH and EVERY one of these steps. The number of <step> tags within your <step_verification> section MUST be exactly {generator_step_count}.
2. You MUST analyze the step and provide YOUR OWN verification reasoning - DO NOT copy the original solution text.
3. Each verification <step> must end with EXACTLY ONE judgement: either \\boxed{{CORRECT}} or \\boxed{{INCORRECT}}.
4. Your final verification within <final_verification> must judge whether the overall solution and final answer are correct.
5. You MUST output ONLY the content within the <step_verification> and <final_verification> tags. Do NOT output anything else.

Your Verification:
"""
