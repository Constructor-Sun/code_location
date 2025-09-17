import os
import json
import subprocess
import difflib
from typing import Dict, List, Union
from pydantic import BaseModel, Field
# _original_get_call_from_func_element = code2flow.python.get_call_from_func_element

# def patched_get_call_from_func_element(func):
#     actual_type = type(func)
#     expected_types = (ast.Attribute, ast.Name, ast.Subscript, ast.Call)
    
#     if actual_type not in expected_types:
#         print(f"Unexpected func type: {actual_type}")
#         print(f"Func element: {ast.dump(func)}")
#         return None
    
#     return _original_get_call_from_func_element(func)

class GetFunctionsInput(BaseModel):
    func_paths: str = Field(
        ...,
        description="""
            List of function paths to retrieve from the corpus dictionary.
            Can be a list of function paths. Elements in this list should NOT just end with files or directories.
        """,
        examples=[
            '{"func_paths": ["sklearn/base.py/DensityMixin/score", "build_tools/github/vendor.py/main"]}'
            ]
    )

class GetFileInput(BaseModel):
    file_path: str = Field(
        ...,
        description="""
            Get all functions in this file. 
            This will match all functions whose paths start with this file path.
            Should not include specific function name.
        """,
        examples=[
            '{"file_path": "build_tools/github/vendor.py"}'
            ]
    )

class ListFunctionDirectoryInput(BaseModel):
    file_or_module_path: str = Field(
        ...,
        description="""
            List all function names in this file or module. 
            This will match all paths whose start with this file or module path.
        """,
        examples=[
            '{"file_or_module_path": "sklearn/pipeline.py"}', '{"file_or_module_path": "build_tools/github/Classifier"}'
            ]
    )

class GetCallGraphInput(BaseModel):
    params: str = Field(
        ...,
        description="""
            JSON string with parameters: {'scopes': [], 'target_function': ''}.
            Scopes: List[str]
                List of scope directories to analyze for the call graph.
                Only accept directory path. DO NOT enter py file name.
                Must be in the form: directory_path
            target_function: str
                The target function name to generate the call graph for.
                Must be in the form: function_name or class_name.function_name.
                DO NOT input class_name only
        """,
        examples=[
            '{"scopes": ["sklearn/", "sklearn/utils/"], "target_function": "get_parser"}',
            '{"scopes": ["models/", "controllers/"], "target_function": "User.create"}',
        ]
    )

def get_corpus(test_dir: str, dataset: str, instance_id: str) -> Dict[str, str]:
    if '-function_' in instance_id:
        corpus_path = os.path.join(test_dir, instance_id, "corpus.jsonl")
    else:
        corpus_path = os.path.join(test_dir, dataset + '-function_' + instance_id, "corpus.jsonl")
    corpus_dict = {}
    # convert corpus to python dicts. key: func_path, value: func_content
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            _id = data.get("_id")
            text = data.get("text")
            
            if _id is not None and text is not None:
                corpus_dict[_id] = text
    return corpus_dict

def get_functions(corpus_dict: Dict[str, str], func_paths: str) -> List[str]:
    try:
        # Parse the JSON string, expecting an object with 'func_paths' key
        input_data = json.loads(func_paths)
        if not isinstance(input_data, dict) or "func_paths" not in input_data:
            return "Input must be a valid JSON sting with a 'func_paths' key"
        paths_list = input_data["func_paths"]
        if not isinstance(paths_list, list):
            return "'func_paths' value must be a list"
    except json.JSONDecodeError as _:
        return "JSON failed to load. Please check the format."

    result = []
    for path in paths_list:
        if not isinstance(path, str):
            result.append(f"Invalid path: {path}. Paths must be strings.")
            continue
        if path in corpus_dict:
            result.append(corpus_dict[path])
        else:
            similar_paths = difflib.get_close_matches(path, corpus_dict.keys(), n=3, cutoff=0.6)
            if similar_paths:
                suggestions = " or ".join(f"'{p}'" for p in similar_paths)
                result.append(f"{path} not found! Do you mean {suggestions}? Ensure the path points to a specific function, e.g., 'sklearn/base.py/DensityMixin/score'.")
            else:
                result.append(f"{path} not found! Ensure the path points to a specific function, e.g., 'sklearn/base.py/DensityMixin/score'.")
    return result

def get_file(corpus_dict: Dict[str, str], file_path: str) -> List[str]:
    try:
        # Parse the JSON string, expecting an object with 'file_path' key
        input_data = json.loads(file_path)
        if not isinstance(input_data, dict) or "file_path" not in input_data:
            return "Input must be a valid JSON sting with a 'file_path' key"
        path = input_data["file_path"]
        if not isinstance(path, str):
            return "'file_path' value must be a string."
    except json.JSONDecodeError as _:
        return "JSON failed to load. Please check the format."

    result = [corpus_dict[func_path] for func_path in corpus_dict.keys()
              if func_path.startswith(path)]
    if not result:
        result.append(f"No functions found for file path '{path}'. Please check if the file path exists in the corpus.")
    return result

# Warning: this should not be used since it lists too many names and confuse the agent
def list_function_directory(corpus_dict: Dict[str, str], file_or_module_path: str) -> List[str]:
    try:
        input_data = json.loads(file_or_module_path)
        if not isinstance(input_data, dict) or "file_or_module_path" not in input_data:
            return "Input must be a valid JSON sting with a 'file_or_module_path' key"
        path = input_data["file_or_module_path"]
        if not isinstance(path, str):
            return "'file_or_module_path' value must be a string."
    except json.JSONDecodeError as _:
        return "JSON failed to load. Please check the format."

    result = [func_path for func_path in corpus_dict.keys()
              if func_path.startswith(path)]
    if not result:
        result.append(f"No functions found for file path '{path}'.")
    return result

def transform_graph(call_graph):
    graph_data = call_graph["graph"]
    
    # node uid -> name
    nodes = graph_data["nodes"]
    node_name_map = {}
    for node_id, node_info in nodes.items():
        node_name_map[node_id] = node_info["name"]
    
    # edges: change uid to name
    transformed_edges = []
    for edge in graph_data["edges"]:
        transformed_edge = {
            "source": node_name_map[edge["source"]],
            "target": node_name_map[edge["target"]]
        }
        transformed_edges.append(transformed_edge)
    
    # create a new json
    result = {
        "directed": graph_data["directed"],
        "edges": transformed_edges
    }
    
    return result

# def _get_call_graph(instance_path: str, scopes: List[str], target_function: str) -> Dict[str, str]:
#     targets = [os.path.join(instance_path, scope) for scope in scopes]
#     modified_targets = []
#     for target in targets:
#         if target.endswith('.py'):
#             if '/' in target:
#                 dir_part = target.rsplit('/', 1)[0]
#                 modified_targets.append(f"{dir_part}/*.py")
#             else:
#                 modified_targets.append("*.py")
#         else:
#             target = target if target.endswith('/') else target + '/'
#             modified_targets.append(target + '*.py')
#     modified_targets = " ".join(modified_targets)
#     code2flow_cmd = f"code2flow {modified_targets} --target-function {target_function} --upstream-depth=1 --downstream-depth=5 --output tmp/call_graph.json"
#     print("code2flow_cmd: ", code2flow_cmd)

#     try:
#         # TODO: sometimes it cannot find the existing node, strange
#         _ = subprocess.run(
#             code2flow_cmd,
#             shell=True,
#             capture_output=True,
#             text=True,
#             check=True
#         )
#         with open("tmp/call_graph.json", "r") as file:
#             call_graph = json.load(file)
#             simple_call_graph = transform_graph(call_graph)
#             with open('tmp/transformed_call_graph.json', 'w', encoding='utf-8') as f:
#                 json.dump(simple_call_graph, f, indent=2, ensure_ascii=False)
#             return simple_call_graph
#     except subprocess.CalledProcessError as e:
#         # print(f"Error running code2flow: {e}")
#         # print(f"stderr: {e.stderr}")
#         # TODO: fix it using: 1. use list_function_directory; 2. do not limit target_function
#         try:
#             fixed_cmd = f"code2flow {modified_targets} --output tmp/call_graph.json"
#             _ = subprocess.run(
#                 fixed_cmd,
#                 shell=True,
#                 capture_output=True,
#                 text=True,
#                 check=True
#             )
#             with open("tmp/call_graph.json", "r") as file:
#                 call_graph = json.load(file)
#                 simple_call_graph = transform_graph(call_graph)
#                 with open('tmp/transformed_call_graph.json', 'w', encoding='utf-8') as f:
#                     json.dump(simple_call_graph, f, indent=2, ensure_ascii=False)
#                 return f"""Cannot find {target_function} in {modified_targets}.
#                             Instead, this tool reveal all call graphs in target_function:
#                             """ + str(simple_call_graph)
#         except subprocess.CalledProcessError as e2:
#             # print(f"Error running code2flow: {e}")
#             # print(f"stderr: {e.stderr}")
#             return "Failed to geneate call graph. Please check if file or function exists."
#     except FileNotFoundError as e:
#         raise RuntimeError(
#             "code2flow not found. Please install with: pip install code2flow"
#             ) from e
    
# def get_call_graph(instance_path: str, params: str) -> Dict[str, str]:
#     try:
#         params_dict = json.loads(params)
#         return _get_call_graph(
#             instance_path,
#             params_dict['scopes'],
#             params_dict['target_function']
#         )
#     except json.JSONDecodeError:
#         return "Error: Invalid JSON format. Please provide valid JSON string."
#     except KeyError as e:
#         return f"Error: Missing required parameter: {e}"
#     except Exception as e:
#         return f"Error: {str(e)}"
    
def _get_call_graph(instance_path: str, scopes: List[str], target_function: str) -> Dict[str, str]:
    targets = [os.path.join(instance_path, scope) for scope in scopes]
    modified_targets = []
    for target in targets:
        if target.endswith('.py'):
            if '/' in target:
                dir_part = target.rsplit('/', 1)[0]
                modified_targets.append(f"{dir_part}/*.py")
            else:
                modified_targets.append("*.py")
        else:
            target = target if target.endswith('/') else target + '/'
            modified_targets.append(target + '*.py')
    modified_targets = " ".join(modified_targets)
    code2flow_cmd = f"code2flow {modified_targets} --target-function {target_function} --upstream-depth=1 --downstream-depth=5 --output tmp/call_graph.json"
    print("code2flow_cmd: ", code2flow_cmd)

    try:
        # TODO: sometimes it cannot find the existing node, strange
        _ = subprocess.run(
            code2flow_cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        with open("tmp/call_graph.json", "r") as file:
            call_graph = json.load(file)
            simple_call_graph = transform_graph(call_graph)
            with open('tmp/transformed_call_graph.json', 'w', encoding='utf-8') as f:
                json.dump(simple_call_graph, f, indent=2, ensure_ascii=False)
            return simple_call_graph
    except subprocess.CalledProcessError as e:
        # print(f"Error running code2flow: {e}")
        # print(f"stderr: {e.stderr}")
        # TODO: fix it using: 1. use list_function_directory; 2. do not limit target_function
        try:
            fixed_cmd = f"code2flow {modified_targets} --output tmp/call_graph.json"
            _ = subprocess.run(
                fixed_cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )
            with open("tmp/call_graph.json", "r") as file:
                call_graph = json.load(file)
                simple_call_graph = transform_graph(call_graph)
                with open('tmp/transformed_call_graph.json', 'w', encoding='utf-8') as f:
                    json.dump(simple_call_graph, f, indent=2, ensure_ascii=False)
                return f"""Cannot find {target_function} in {modified_targets}.
                            Instead, this tool reveal all call graphs in target_function:
                            """ + str(simple_call_graph)
        except subprocess.CalledProcessError as e2:
            # print(f"Error running code2flow: {e}")
            # print(f"stderr: {e.stderr}")
            return "Failed to geneate call graph. Please check if file or function exists."
    except FileNotFoundError as e:
        raise RuntimeError(
            "code2flow not found. Please install with: pip install code2flow"
            ) from e
    
def get_call_graph(instance_path: str, params: str) -> Dict[str, str]:
    try:
        params_dict = json.loads(params)
        return _get_call_graph(
            instance_path,
            params_dict['scopes'],
            params_dict['target_function']
        )
    except json.JSONDecodeError:
        return "Error: Invalid JSON format. Please provide valid JSON string."
    except KeyError as e:
        return f"Error: Missing required parameter: {e}"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    corpus = get_corpus("datasets", "swe-bench-lite", "scikit-learn__scikit-learn-25500")
    test_funcs = False
    test_file = False
    test_call_graph = True
    if test_funcs:
        example_func_path = [
            "build_tools/generate_authors_table.py/get",
            "build_tools/generate_authors_table.py/key",
            "scikit-learn/setup.py/CleanCommand/run"
        ]
        funcs = get_functions(corpus, example_func_path)
        print("funcs: ", funcs)
    if test_file:
        file_path = "build_tools/generate_authors_table.py"
        funcs = get_file(corpus, file_path)
        print("funcs: ", len(funcs))
    if test_call_graph:
        instance_path = os.path.join("swe-bench-lite", "scikit-learn__scikit-learn-14983")
        params = '{"scopes": ["sklearn/model_selection/_split.py"], "target_function": "_RepeatedSplits"}'
        result = get_call_graph(instance_path, params)
        print("result: ", result)

if __name__ == "__main__":
    main()