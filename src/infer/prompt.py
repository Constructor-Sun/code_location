FIRST_PROMPT = """
Given the following GitHub problem description, your objective is to localize the specific methods or module-level functions that need modification or contain key information to resolve the issue. You have initial candidate functions for reference.
- Issue Description: {query}
- Initial Candidates (preds): {preds}

Follow these steps to localize the issue:
Step 1: Categorize Issue Description and Find Entry Points
 - Classify the issue description into the following categories:
    Problem description, error trace, code to reproduce the bug, and additional context.
 - Identify main entry points triggering the issue in the description.
 - Prioritize the inital code candidates (preds): they are retrieved as root cause methods or module-level functions with high probability.

Step 2: Explore Classes and methods
- Explore the repo to familiarize yourself with its structure. (TOOL: list_function_directory, get_file)
- Explore classes and its methods, from the entry points in step 1 (LOOP) (TOOL: get_class)
    - LOOP Start
    - Operation in loop: Use get_class to get the context of one class related to the issue.
    - LOOP ENDS: You have explored 4~5 classes or files and analyzed their methods.
- Pay special attention to distinguishing between classes with similar names using context and described execution flow.

Step 3: Expand Your Thoughts
- Assume the issue involves classes from Step 2 to trace additional bug paths.
  - LOOP: For each class (4-5 total), invoke get_functions (3~4 methods within this class) to analyze issue propagation within it.
  - END LOOP: All Step 2 classes covered.

Step 4: Identify Target methods for Modification
- Determine those methods or module-level functions that require changes to fully resolve the issue
    - Try consider methods in different classes in Step 2, with each about 3 steps
    - You should not list too much function within single class since it is not a architecture issue

## Tool calls Input**:
- get_call_graph Format: '{{"target_function": ''}}'
- get_functions Format: '{{"func_paths": []}}'
- get_class Format: '{{"class_path": ''}}'
- list_function_directory Format : '{{"file_path": ''}}'
- get_file Format: '{{"target_function": ''}}'

## Final Answer Format:
- You should return a valid JSON string with key "updated_methods", whose value should include exactly 10 methods or module-level functions
- Ranked in order of likelihood that you consider it to be the root cause.
- Final Answer:{{
    "updated_methods": ["Must", "be", "function", "names", "not", "class", "names"],
}}
"""

SECOND_PROMPT = """
IMPORTANT DEFINITIONS:
- Callee: A function/method invoked BY the candidate (downstream/outgoing calls).
- Caller: A function/method that invokes the candidate (upstream/incoming calls). DO NOT analyze or include callers EVER.

GOAL: Identify methods to modify by following the CALL CHAIN DOWNSTREAM from candidates (callees, then callees of callees, etc.). Prioritize methods in these files: {file}. You MAY extend to functions in other files ONLY if they are directly called by candidates or their callees (proven via tool traces). Always aim for at least 4 methods; if fewer found, select the best available and note in reasoning.

Given the following GitHub problem description, your objective is to use initial candidates to fix this issue.
- Issue Description: {query}
- Candidates: {candidates}

Follow these steps, ensuring DOWNSTREAM-ONLY (callee) analysis. Never go upstream to callers.

Step 1: Downstream Dependency Analysis (Callee-Only Chain)
- Start with Level 0: The candidates themselves.
- For each at current level, analyze ONLY WHAT IT CALLS (callees):
  - Direct function calls within the methods.
  - Class instantiations and method invocations.
  - Return types and their dependencies (if callable).
- Use tools to trace: Call get_functions or get_file to verify callees and their files.
- Build a chain level-by-level (max depth 3 to avoid loops):
  - Level 1: Direct callees of candidates.
  - Level 2: Callees of Level 1.
  - Etc.
- Discard ANYTHING that traces back to callers. Example: If A calls B (B is callee of A), do NOT include functions that call A.
- If a callee is outside {file}, include it ONLY if tool-verified as in the chain.
- Reasoning Trace: List the chain like: "Level 0: candidate1 -> Level 1: calleeX (in fileY) -> Level 2: calleeZ (in fileW)"
- Termination: Stop at depth 3 or if no new callees. If chain is short (<4 total methods), reuse from earlier levels or note "limited chain" but still output 4 (e.g., duplicates if needed).

Step 2: Identify Feasible Methods to Be Modified
- From the callee chain, select the most appropriate methods for modification (at least 4).
- Prioritize those in {file}; include external only if chain-linked.
- All must be methods or module-level functions (complete paths). Never return class names.
- Validation: For each selected, confirm it's a callee (not caller) via chain trace. If not, discard and replace.

Step 3: Anti-Loop and Debug Check
- If stuck (e.g., no callees in {file}), broaden to chain-extended but stop after 2 tool calls max per level.
- Verify: No callers included? Chain depth <=3? At least 4 outputs?
- Example Correct Chain: Candidate: funcA in file1.py calls funcB in file1.py, which calls funcC in file2.py. Methods: ["file1.py:funcA", "file1.py:funcB", "file2.py:funcC", "file1.py:someOtherCallee"]
- Example Incorrect (DO NOT): Include funcD that calls funcA (that's a caller).

## Tool Calls Input:
- get_functions Format: '{{"func_paths": []}}'
- get_class Format: '{{"class_path": ''}}'
- get_file Format: '{{"target_function": ''}}'
- Use tools sparingly to verify chains; batch if possible (e.g., multiple func_paths in one call).

## Final Answer Format:
- Return ONLY a valid JSON string with "main_idea" and "methods_tobe_modified".
- "methods_tobe_modified" MUST include at least 4 methods with complete paths, from the callee chain ONLY.
- Final Answer:{{
    "main_idea": "",
    "methods_tobe_modified": []
}}
"""

# SECOND_PROMPT = """
# IMPORTANT CONSTRAINT: All analysis and modifications MUST be limited to methods or module-level functions within these files ONLY: {file}. You MUST NOT reference, analyze, or suggest ANY methods from other files. If a dependency leads outside {file}, ignore it and focus solely on elements within {file}.

# Given the following GitHub problem description, your objective is to use initial candidates to fix this issue.
# - Issue Description: {query}
# - Candidates: {candidates}

# Now, follow these steps to help you fix the issue, ensuring EVERY STEP adheres to the constraint above (methods/functions ONLY from {file}):

# Step 1: Downstream Dependency Analysis (Limited to {file})
# - For each candidate function, analyze WHAT IT CALLS (callee analysis), but ONLY trace calls to functions/methods that exist within {file}.
# - Trace outgoing calls: functions/methods invoked by the candidates, but discard any that are not in {file}.
#     - Direct function calls within candidate methods (only if callee is in {file})
#     - Class instantiations and method invocations (only if class/method is defined in {file})  
#     - Return types and their dependencies (only if types/dependencies are handled in {file})
# - DO NOT analyze who calls the candidates (caller analysis). If no valid callees in {file}, note that and proceed.

# Step 2: Identify Feasible Methods to Be Modified
# - Assume the root cause resides within the same files as the original candidates ({file}), then find the most appropriate modification methods.
# - All methods or module-level functions MUST be in these files ONLY: {file}. If a potential method is not in {file}, discard it immediately and select an alternative from {file}.
# - You MUST return methods or module-level functions ONLY. Never return a class name, even if original candidates include class names.
# - Aim for at least 4 methods, all with complete paths, exclusively from {file}.

# Step 3: Validation Check
# - Review your identified methods: For each one, confirm it is defined in {file}. If ANY method is from outside {file}, replace it with a valid one from {file} or explain why no alternative exists (but still provide at least 4 valid ones).

# Example of Correct Output:
# Suppose {file} = ['file1.py', 'file2.py'], and methods must be from there.
# Correct "methods_tobe_modified": ["file1.py:func_a", "file1.py:func_b", "file2.py:method_c", "file2.py:func_d"]
# Incorrect (do NOT do this): ["external.py:func_x"] - this would violate the rule.

# ## Tool Calls Input:
# - get_functions Format: '{{"func_paths": []}}'
# - get_class Format: '{{"class_path": ''}}'
# - get_file Format: '{{"target_function": ''}}'

# ## Final Answer Format:
# - You should only return a valid JSON string with keys "main_idea" and "methods_tobe_modified".
# - "methods_tobe_modified" MUST include at least 4 methods with complete paths, ALL of them from {file} ONLY. Double-check this before outputting.
# - Final Answer:{{
#     "main_idea": "",
#     "methods_tobe_modified": []
# }}
# """

REACT_PROMPT = '''Answer the following questions as best you can. 
You have access to the following tools:
{tools}
**Tool calls failure handing:** If a tool fails, follow its returned suggestions to get your results.
Use the following format:

Question: the input question you must answer
Thought: Explain your thinking step-by-step. You're encouraged to give long thought.
Action: the action to take, should be one of [{tool_names}]. You should only enter tool's name, e.g., get_file
Action Input: the input to the action, e.g., '{{"func_paths": []}}'
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times before you give your final answer)
Final Answer: I now know the final answer... The final answer is listed in the JSON string: ... 
(After you give this JSON string, you must STOP. You should never repeat the same JSON string.)

Begin!

Question: {input}
Thought: {agent_scratchpad}
'''