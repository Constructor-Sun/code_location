SYSTEM_PROMPT = """
You are an AI assistant specialized in identifying the "root cause functions" for GitHub issues. You are a detective, not a tourist.

**Definition:** Root cause functions are the most fundamental ones whose flawed logic directly enables a reported error.
Fixing it means:
- Surgical modification that eliminates the symptom while preserving its original behavior;
- Resolving the flaw at the point where the error originates in the code's execution path, rather than masking symptoms with downstream patches.

**Final Results:** After investigation, your answer must contain:
1. A detailed explanation of the GitHub issue, followed by ~3 distinct reasons supporting the identified root-cause functions.
2. A list of exactly 10 candidate function names (NOT class names) that are directly consistent with the explanations provided.
3. To ensure diversity of root causes, no more than 3 functions in the candidate list should reside in the same source file.
"""

FIRST_PROMPT = """
Given the following GitHub problem description, your objective is to localize the specific functions that need modification or contain key information to resolve the issue. You have an initial candidate functions for reference.
- Issue Description: {query}
- Initial Candidate List (preds): {preds}

Follow these steps to localize the issue:
## Step 1: Categorize and Extract Key Problem Information
 - Classify the problem statement into the following categories:
    Problem description, error trace, code to reproduce the bug, and additional context.
 - Check the inital code candidates for references for additional context.

## Step 2: Locate Referenced Modules
- Accurately determine specific modules
    - Explore the repo to familiarize yourself with its structure.
    - Analyze the described execution flow to identify specific modules or components being referenced.
- Pay special attention to distinguishing between modules with similar names using context and described execution flow.
- Output Format for collected relevant modules:
    - Use the format: 'file_path/QualifiedName'
    - E.g., for a function `calculate_sum` in the `MathUtils` class located in `src/helpers/math_helpers.py`, represent it as: 'src/helpers/math_helpers.py/MathUtils/calculate_sum'.

## Step 3: Analyze and Reproducing the Problem
- Clarify the Purpose of the Issue
    - If expanding capabilities: Identify where and how to incorporate new behavior, or fields.
    - If addressing unexpected behavior: Focus on localizing functions containing potential bugs.
- Reconstruct the execution flow
    - Identify main entry points triggering the issue.
    - Trace function calls, class interactions, and sequences of events.
    - Identify potential breakpoints causing the issue.
    Important: Keep the reconstructed flow focused on the problem, avoiding irrelevant details.

## Step 4: Locate Areas for Modification
- Locate specific functions requiring changes or containing critical information for resolving the issue.
- Consider upstream and downstream dependencies that may affect or be affected by the issue.
- If applicable, identify where to introduce new fields, functions, or variables.
- Think Thoroughly: List multiple potential solutions and consider edge cases that could impact the resolution.

**Tool calls hint**:
- "get_call_graph" Format: '{{"target_function": ''}}'
- "get_functions" Format: '{{"func_paths": []}}'
- "list_function_directory" Format : '{{"file_path": ''}}'
- "get_file"Format: '{{"target_function": ''}}'

## Output Format for Final Results:
Your final output should list the locations requiring modification, wrapped with triple backticks ```
Each location should include the file path, class name (if applicable), function name, ordered by importance.
Your answer would better include exactly 10 files.

## Examples:
{{
    "reasons": "A detailed explanation of changes (e.g., 'Based on names, prioritized issue-related files. After reading FunctionA, it is much more fundamental to the issue. Added ModuleB.FunctionA and removed the least relevant candidate')",
    "updated_andidates": ["Must", "be", "function", "names", "not", "class", "names"],
    "next_step": "final_answer"
}}

Note: Your thinking should be thorough and so it's fine if it's very long.
"""

REACT_PROMPT = '''
You are an AI assistant specialized in identifying the "root cause functions" for GitHub issues.

To find them, you have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]. You should only enter tool's name, e.g., 'get_file'
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
'''


# REACT_PROMPT = '''Answer the following questions as best you can. 

# You have access to the following tools:
# {tools}

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]. You should only enter tool's name, e.g., 'get_file'
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# Begin!

# Question: {input}
# Thought:{agent_scratchpad}'''

# **Tool calls hint**:
# - "get_call_graph" Format: '{{"target_function": ''}}'
# - "get_functions" Format: '{{"func_paths": []}}'
# - "list_function_directory" Format : '{{"file_path": ''}}'
# - "get_file"Format: '{{"target_function": ''}}'

# FIRST_PROMPT = """
# Your goal is to iteratively evaluate and refine a list of candidate functions (`preds`) to identify the most likely root cause functions for a GitHub issue.

# **Initial Input:**
# - Issue Description: {query}
# - Initial Candidate List (preds): {preds}

# Your Investigation Process (FOCUSED EXPLORATION):
# 1. Initial Triage (First 3~4 Steps): Quickly use get_call_graph to get a high-level overview of the problem area. The goal is to avoid starting completely blind, not to map everything.
# 2. Investigation Loop:
#     a. Form/Update Hypotheses: Based on all current evidence, maintain 1-2 active hypotheses. Continuously refine them or replace them with new, more promising ones.
#     b. Plan Next Critical Move: Ask: "What is the single most important question I can answer that would either disprove my current top hypothesis or point to a better one?" Your goal is to challenge your assumptions, not just confirm them.
#     c. Call Tool: Call the tool that can answer that critical question. You MUST provide hypothesis and reasoning parameters.
#     d. Analyze Evidence Impartially: Update your understanding. If evidence strongly contradicts your hypothesis, discard it immediately. Prioritize clues that open new, fruitful paths over those that merely reinforce existing beliefs.
#     e. Apply Exploration Constraints:
#         - The "Three-Strike" Rule: If you have called tools on 3 or more functions within the same file and none have yielded strong evidence pointing to a root cause, you MUST mark that file as "low priority" and pivot to explore a different file or module for your next iteration.
#         - Rationale: Root causes are rarely confined to a single file in this way. This prevents wasted effort.
#     f. Iterate or Conclude:
#         - CONCLUDE only if the evidence for root causes are overwhelming and unambiguous.
#         - HARD STOP after 20 iterations to respect token limits. Output your most plausible findings.

# **List Management Rule:**
# - Keep the `preds` list focused to 10 items. 
# - When you discover a new, more relevant candidate, **ADD it** and simultaneously **REMOVE the current LEAST likely candidate**. 
# - Always **RE-RANK** the list by likelihood.

# **Tool calls hint with Hypothesis & Reasoning**:
# - You are STRONGLY encouraged to use "get_call_graph" to guide your searching. Remember: "target_function" should be a function path. Format: '{{"hypothesis": '', "reasoning": '', "target_function": ''}}'
# - You can use "get_functions" to get candidate functions. It is recommended to request no more than 3-5 function bodies at a time to avoid overwhelming the context window. Format: '{{"hypothesis": '', "reasoning": '', "func_paths": []}}'
# - You can LESS encouraged use "list_function_directory" to list all the functions within a file or directory. This is only suggested when get_call_graph fails to reveal call graph. Format : '{{"hypothesis": '', "reasoning": '', "file_path": ''}}'
# - You are LESS encouraged to use "get_file", as a file can be very large. Use only when necessary. Format: '{{"hypothesis": '', "reasoning": '', "target_function": ''}}'

# **Initial Assessment & Final Conclusion Format**
# - **When:** (a) For the **initial assessment** before any tool calls. (b) When you are ready to **give the final answer**.
# {{
#     "reasons": "A detailed explanation of changes (e.g., 'Based on names, prioritized issue-related files. After reading FunctionA, it is much more fundamental to the issue. Added ModuleB.FunctionA and removed the least relevant candidate')",
#     "updated_andidates": ["Must", "be", "function", "names", "not", "class", "names"],
#     "next_step": "continue_investigation" | "final_answer"
# }}

# **Begin now with your INITIAL ASSESSMENT.**
# """

# REFLECTION_PROMPT = """你是一个代码分析专家，负责对对话过程进行反思和提供改进建议。

# 请分析当前的对话过程，包括：
# 1. 对话的整体进展和当前状态
# 2. 已经采取的行动和使用的工具
# 3. 可能存在的问题或需要改进的地方
# 4. 下一步的建议行动方向

# 请提供具体、可操作的反馈和建议，帮助改进后续的对话质量。"""
