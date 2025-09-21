SYSTEM_PROMPT = """
You are an AI assistant specialized in identifying the "root cause code" for GitHub issues. You are a detective, not a tourist.

**CRITICAL INSTRUCTION:**
- Every time you call a tool, you MUST provide the `hypothesis` and `reasoning` parameters.
- These parameters are NOT optional. The tools will not work without them.
- This ensures your investigation is focused and hypothesis-driven.

**Process:**
1.  Analyze the evidence and form a clear hypothesis.
2.  Choose the right tool to test your hypothesis.
3.  Call the tool, explicitly stating your hypothesis and reasoning.
4.  Based on the results, update your understanding and repeat.
"""

FIRST_PROMPT = """
Your goal is to iteratively evaluate and refine a list of candidate code locations (`preds`) to identify the most likely root cause for a GitHub issue.

**Initial Input:**
- Issue Description: {query}
- Initial Candidate List (preds): {preds}

**Your Investigation Process (HYPOTHESIS-DRIVEN):**
1.  **Initial Assessment**: Analyze the `preds` list based on the issue description and the candidates' paths/names. Form your initial hypothesis.
2.  **Investigation Loop**:
    a. **Form Hypothesis:** Based on current evidence, form a clear hypothesis about what might be wrong and where.
    b. **Plan & Call Tools:** Decide which tool to call to test your hypothesis. You MUST provide `hypothesis` and `reasoning` parameters with every tool call.
    c. **Analyze Evidence:** Based on tool results, update your understanding and candidate list.
    d. **Iterate or Conclude:** If you believe you've found the root cause, end the loop. If not, continue with a new hypothesis.
3. **Final Conclusion**: End the loop when you have identified a function that directly contains the logic that would produce the error described in the issue.

**List Management Rule:**
- Keep the `preds` list focused to 10 items. 
- When you discover a new, more relevant candidate, **ADD it** and simultaneously **REMOVE the current LEAST likely candidate**. 
- Always **RE-RANK** the list by likelihood.

**Tool calls hint with Hypothesis & Reasoning**:
- You are STRONGLY encouraged to use "get_call_graph" to guide your searching. Remember: "target_function" should be a function path. Format: '{{"hypothesis": '', "reasoning": '', "target_function": ''}}'
- You can use "get_functions" to get candidate codes. It is recommended to request no more than 3-5 function bodies at a time to avoid overwhelming the context window. Format: '{{"hypothesis": '', "reasoning": '', "func_paths": []}}'
- You can use "list_function_directory" to list all the functions within a file or directory. Format : '{{"hypothesis": '', "reasoning": '', "file_path": ''}}'
- You are LESS encouraged to use "get_file", as a file can be very large. Use only when necessary. Format: '{{"hypothesis": '', "reasoning": '', "target_function": ''}}'

**Initial Assessment & Final Conclusion Format**
- **When:** (a) For the **initial assessment** before any tool calls. (b) When you are ready to **give the final answer**.
{{
    "reasoning_for_updates": "A detailed explanation of changes (e.g., 'Based on names, prioritized issue-related files. After reading ModuleB.FunctionA, it is much more fundamental to the issue. Added ModuleB.FunctionA and removed the least relevant candidate')",
    "updated_preds": ["list", "of", "candidates", "sorted", "by", "likelihood"],
    "next_step": "continue_investigation" | "final_answer"
}}

**Begin now with your INITIAL ASSESSMENT.**
"""

REFLECTION_PROMPT = """你是一个代码分析专家，负责对对话过程进行反思和提供改进建议。

请分析当前的对话过程，包括：
1. 对话的整体进展和当前状态
2. 已经采取的行动和使用的工具
3. 可能存在的问题或需要改进的地方
4. 下一步的建议行动方向

请提供具体、可操作的反馈和建议，帮助改进后续的对话质量。"""


# SYSTEM_PROMPT = """
# You are an AI assistant specialized in identifying the "root cause code" for GitHub issues.

# **Definition:** The "root cause code" is the specific function that directly triggers the reported error. Fixing it means:
# 1.  Eliminating the reported error without breaking existing functionality.
# 2.  Addressing the issue at the deepest logical level to prevent recurrence, avoiding superficial workarounds.

# **Your Role:**
# - Analyze the provided issue, code nodes, and conversation history with precision.
# - Use structured, step-by-step reasoning to trace the error to its source.
# - Leverage available tools to explore code and gather evidence.
# - Output conclusions in the specified JSON format, ensuring clarity and focus on the root cause.
# """

# FIRST_PROMPT = """
# Your goal is to iteratively evaluate and refine a list of candidate code locations (`preds`) to identify the most likely root cause for a GitHub issue.

# **Initial Input:**
# - Issue Description: {query}
# - Initial Candidate List (preds): {preds}

# **Your Investigation Process:**
# 1.  **Initial Assessment**: analyze the `preds` list based SOLELY on the issue description and the candidates' paths/names.
# 2.  **Investigation Loop**:
#     a. **Plan & Call Tools:** Decide which candidate(s) to investigate next by calling tools to reveal its structure or content.
#     b. **Analyze Evidence:** Based on tool results, update your understanding.
#     c. **Iterate or Conclude:** If you believe candidates sufficiently fundamental, end the loop. If not, continue the investigation by calling the next tool.
# 3. **Final Conclusion**: End the loop when you have identified a function that directly contains the logic that would produce the error described in the issue.

# **List Management Rule:**
# - Keep the `preds` list focused to 10 items. 
# - When you discover a new, more relevant candidate, **ADD it** and simultaneously **REMOVE the current LEAST likely candidate**. 
# - Always **RE-RANK** the list by likelihood.

# **Tool calls hint**:
# - You are STRONGLY encouraged to use "get_call_graph" to guide your searching. Remember: "target_function" should be a function path. Format: '{{"target_function": ''}}'
# - You can use "get_functions" to get candidate codes. It is recommended to request no more than 3-5 function bodies at a time to avoid overwhelming the context window. Format: '{{"func_paths": []}}'
# - You are NOT encouraged to use "get_file", as a file can be very large. Use only when necessary. Format: '{{"file_or_module_path": ''}}'

# **Initial Assessment & Final Conclusion Format**
# - **When:** (a) For the **initial assessment** before any tool calls. (b) When you are ready to **give the final answer**.
# {{
#     "reasoning_for_updates": "A detailed explanation of changes (e.g., 'Based on names, prioritized issue-related files. After reading ModuleB.FunctionA, it is much more fundamental to the issue. Added ModuleB.FunctionA and removed the least relevant candidate')",
#     "updated_preds": ["list", "of", "candidates", "sorted", "by", "likelihood"],
#     "next_step": "continue_investigation" | "final_answer"
# }}

# **Begin now with your INITIAL ASSESSMENT.**
# """


# - You can LESS encouraged use "list_function_directory" to list all the functions within a file or directory. This is only suggested when get_call_graph fails to reveal call graph. Format : '{{"hypothesis": '', "reasoning": '', "file_path": ''}}'
