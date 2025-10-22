FIRST_PROMPT = """
Given the following GitHub problem description, your objective is to localize the specific methods or module-level functions that need modification or contain key information to resolve the issue. You have initial candidate functions for reference.
- Issue Description: {query}
- Initial Candidates (preds): {preds}

Follow these steps to localize the issue:
Step 1: Categorize Issue Description and Find Entry Points
 - Classify the issue description into the following categories:
    Problem description, error trace, code to reproduce the bug, and additional context.
 - Identify main entry points triggering the issue in the description.

Step 2: Check the inital code candidates (preds)
 - Prioritize them: they are retrieved as root cause methods or module-level functions with high probability. (TOOL: get_functions)
 
Step 3: Explore Classes and methods
- Explore the repo to familiarize yourself with its structure. (TOOL: list_function_directory, get_file)
- Explore classes and its methods, from the entry points in step 1 (LOOP) (TOOL: get_class)
    - LOOP Start
    - Operation in loop: Use get_class to get the context of one class related to the issue.
    - LOOP ENDS: You have explored 4~5 classes or files and analyzed their methods.
- Pay special attention to distinguishing between classes with similar names using context and described execution flow.

Step 4: Expand Your Thoughts
- Assume the issue involves classes from Step 2 to trace additional bug paths.
  - LOOP: For each class (4-5 total), invoke get_functions (3~4 methods within this class) to analyze issue propagation within it.
  - END LOOP: All Step 2 classes covered.

Step 5: Identify Target methods for Modification
- Determine those methods or module-level functions that require changes to fully resolve the issue
    - Try consider methods in different classes in Step 2, with each about 3 steps
    - You should not list too much function within single class since it is not a architecture issue

## Tool calls Input**:
- get_call_graph Format: '{{"target_function": ''}}'
- get_functions Format: '{{"func_paths": []}}'
- get_class Format: '{{"class_path": ''}}'
- list_function_directory Format : '{{"file_path": ''}}'
- get_file Format: '{{"file_path": ''}}'

## Final Answer Format:
- You should return a valid JSON string with key "updated_methods", whose value should include exactly 10 methods or module-level functions
- Final Answer:{{
    "updated_methods": ["Must", "be", "function", "names", "not", "class", "names"],
}}
"""

SECOND_PROMPT = """
IMPORTANT CONSTRAINT: All analysis and modifications MUST be limited to methods or module-level functions within these files ONLY: {scope}. You MUST NOT reference, analyze, or suggest ANY methods from other files. If a dependency leads outside {scope}, ignore it and focus solely on elements within {scope}.

Given the following GitHub problem description, your objective is to use initial candidates to fix this issue.
- Issue Description: {query}
- Candidates: {candidates}

Now, follow these steps to help you fix the issue, ensuring EVERY STEP adheres to the constraint above (methods/functions ONLY from {scope}):

Step 1: Be familiar with {scope}
- Use get_file or get_class to get the full context of {scope}
- Pay attention to each member in {scope} using the following loop. At this stage, you should take a thorough investigation
    - LOOP STARTS: If {scope} is a file, use get_functions to get module-level functions, then use get_class to get classes
    - LOOP ENDS: each member mentioned in Step 1 is aquired

Step 2: Downstream Dependency Analysis (Limited to {scope})
- For each candidate function, analyze WHAT IT CALLS (callee analysis), but ONLY trace calls to functions/methods that exist within {scope}.
- Trace outgoing calls: functions/methods invoked by the candidates, but discard any that are not in {scope}.
    - Direct function calls within candidate methods (only if callee is in {scope})
    - Class instantiations and method invocations (only if class/method is defined in {scope})  
    - Return types and their dependencies (only if types/dependencies are handled in {scope})
- DO NOT analyze who calls the candidates (caller analysis). If no valid callees in {scope}, note that and proceed.

Step 3: Identify Feasible Methods to Be Modified
- Assume the root cause resides within the same files as the original candidates ({scope}), then find the most appropriate modification methods.
- All methods or module-level functions MUST be in these files ONLY: {scope}. If a potential method is not in {scope}, discard it immediately and select an alternative from {scope}.
- You MUST return methods or module-level functions ONLY. Never return a class name, even if original candidates include class names.
- Aim for exactly 4 methods, all with complete paths, exclusively from {scope}.

Step 4: Validation Check
- Review your identified methods: For each one, confirm it is defined in {scope}. If ANY method is from outside {scope}, replace it with a valid one from {scope} or explain why no alternative exists (but still provide exactly 4 valid ones).

Example of Correct Output:
Suppose {scope} = ['file1.py', 'file2.py'], and methods must be from there.
Correct "methods_tobe_modified": ["file1.py:func_a", "file1.py:func_b", "file2.py:method_c", "file2.py:func_d"]
Incorrect (do NOT do this): ["external.py:func_x"] - this would violate the rule.

## Tool Calls Input:
- get_functions Format: '{{"func_paths": []}}'
- get_class Format: '{{"class_path": ''}}'
- get_file Format: '{{"file_path": ''}}'

## Final Answer Format:
- You should only return a valid JSON string with keys "main_idea" and "methods_tobe_modified".
- "methods_tobe_modified" MUST include exactly 4 methods with complete paths, ALL of them from {scope} ONLY. Double-check this before outputting.
- Final Answer:{{
    "main_idea": "",
    "methods_tobe_modified": []
}}
"""

# SECOND_PROMPT = """
# Given the following GitHub problem description, your object is to use initial candidates to fix this issue.
# - Issue Description: {query}

# Now, follow these steps to help you fix the issue:

# Step 1: Be familiar with the structure of {scope}
# - If {scope} ends with .py, use get_file to get the full context; Otherwise use get_class to get the full context;

# Step 2: Be familiar with the context of {scope}
# - Pay attention to each member in {scope} using the following loop. At this stage, you should take a thorough investigation
#     - LOOP STARTS: If {scope} is a file, use get_functions to get module-level functions, then use get_class to get classes; If {scope} is a class, use get_functions to get each methods.
#     - LOOP ENDS: each member mentioned in Step 1 is aquired

# Step 3: Identify feasible methods to be modified
# - Assume the root cause resides within {scope}, then find out the most appropriate modification methods or module-level functions:
# 	- First consider {candidates}
# 	- If methods or module-level functions from Step 2 are more suitable to be modified, then consider them as the most appropriate ones.
# - You must return methods or module-level functions. You should never return a class name even if original candidates are sometimes class names.

# ## Tool calls Input**:
# - get_functions Format: '{{"func_paths": []}}'
# - get_class Format: '{{"class_path": ''}}'
# - get_file Format: '{{"target_function": ''}}'

# ## Final Answer Format:
# - You should only return a valid JSON string with key "main_idea" and "methods_tobe_modified"
# - "methods_tobe_modified" should include at least 4 methods with complete paths, all of them must be in theses files ONLY: {scope}.
# - Final Answer:{{
#     "main_idea": "",
#     "methods_tobe_modified": []
# }}
# """

# REACT_PROMPT = '''Answer the following questions as best you can. 
# You have access to the following tools:
# {tools}
# **Tool calls failure handing:** If a tool fails, follow its returned suggestions to get your results.

# **CRITICAL FORMAT RULES:**
# - After each "Action Input:", you MUST STOP and wait for the Observation from the environment.
# - NEVER generate "Observation:" yourself - this will be provided by the environment.
# - If you generate an Observation, the tool will fail.

# Use the following format:
# 1. Question: the input question you must answer
# 2. LOOP: think which tool you should call next and analyze current situation. This loop can repeat N times before you give your final answer.
#     - What you give:
#         - Thought: Explain your thinking step-by-step based on *Question* and *Observation*. You're encouraged to give long thought.
#         - Action: tool calls, should be one of [{tool_names}]. You should only enter tool's name, e.g., get_file
#         - Action Input: a valid JSON string containing parameters of tool calls, e.g., '{{"func_paths": []}}'.
#         *** STOP HERE after Action Input ***
#     - What the environment returns:
#         - Observation: the result of tool calls
# 3. Final Answer: I now know the final answer... The final answer is listed in the JSON string: ... 
# (After you give this JSON string, you MUST STOP. You should never repeat the same JSON string.)

# Begin!

# Question: {input}
# Thought: {agent_scratchpad}
# '''

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

# SECOND_PROMPT = """
# IMPORTANT CONSTRAINT: All analysis and modifications MUST be limited to methods or module-level functions within these files ONLY: {scope}. You MUST NOT reference, analyze, or suggest ANY methods from other files. If a dependency leads outside {scope}, ignore it and focus solely on elements within {scope}.

# Given the following GitHub problem description, your objective is to use initial candidates to fix this issue.
# - Issue Description: {query}
# - Candidates: {candidates}

# Now, follow these steps to help you fix the issue, ensuring EVERY STEP adheres to the constraint above (methods/functions ONLY from {scope}):

# Step 1: Be familiar with {scope}
# - Use get_file or get_class to get the full context of {scope}
# - Try analyze each member in the {scope}

# Step 2: Downstream Dependency Analysis (Limited to {scope})
# - For each candidate function, analyze WHAT IT CALLS (callee analysis), but ONLY trace calls to functions/methods that exist within {scope}.
# - Trace outgoing calls: functions/methods invoked by the candidates, but discard any that are not in {scope}.
#     - Direct function calls within candidate methods (only if callee is in {scope})
#     - Class instantiations and method invocations (only if class/method is defined in {scope})  
#     - Return types and their dependencies (only if types/dependencies are handled in {scope})
# - DO NOT analyze who calls the candidates (caller analysis). If no valid callees in {scope}, note that and proceed.

# Step 3: Identify Feasible Methods to Be Modified
# - Assume the root cause resides within the same files as the original candidates ({scope}), then find the most appropriate modification methods.
# - All methods or module-level functions MUST be in these files ONLY: {scope}. If a potential method is not in {scope}, discard it immediately and select an alternative from {scope}.
# - You MUST return methods or module-level functions ONLY. Never return a class name, even if original candidates include class names.
# - Aim for at least 4 methods, all with complete paths, exclusively from {scope}.

# Step 4: Validation Check
# - Review your identified methods: For each one, confirm it is defined in {scope}. If ANY method is from outside {scope}, replace it with a valid one from {scope} or explain why no alternative exists (but still provide at least 4 valid ones).

# Example of Correct Output:
# Suppose {scope} = ['file1.py', 'file2.py'], and methods must be from there.
# Correct "methods_tobe_modified": ["file1.py:func_a", "file1.py:func_b", "file2.py:method_c", "file2.py:func_d"]
# Incorrect (do NOT do this): ["external.py:func_x"] - this would violate the rule.

# ## Tool Calls Input:
# - get_functions Format: '{{"func_paths": []}}'
# - get_class Format: '{{"class_path": ''}}'
# - get_file Format: '{{"target_function": ''}}'

# ## Final Answer Format:
# - You should only return a valid JSON string with keys "main_idea" and "methods_tobe_modified".
# - "methods_tobe_modified" MUST include at least 4 methods with complete paths, ALL of them from {scope} ONLY. Double-check this before outputting.
# - Final Answer:{{
#     "main_idea": "",
#     "methods_tobe_modified": []
# }}
# """